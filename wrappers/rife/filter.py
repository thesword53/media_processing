import numpy as np
import torch
import os
from torch.nn import functional as F
from filter import Filter, filters
from train_log.RIFE_HDv3 import Model
from model.pytorch_msssim import ssim_matlab
import flogging

current_dir = "wrappers"+os.path.sep+"rife"

def argsToDict(argsStr):
    argList = {}
    if len(argsStr) > 0:
        argsStrList = argsStr.split(';')
        for arg in argsStrList:
            elts = arg.split('=', 1)
            if len(elts) >= 2:
                argList[elts[0]]=elts[1]
            else:
                return None
    return argList

class RIFEFilter(Filter):
    def __init__(self, args, fp16=True):
        self.log = flogging.FilterLogging(args["loglevel"], "RIFE")
        self.args = argsToDict(args["user"])
        self._getVideoInfo(args)
        self.fp16 = fp16
        self.lastframe = None
        self.useHighSsim = True
        args["reader"].dtype = np.float32
        args["reader"].batchSize = 1
        args["reader"]._4D = True

        self.scale = 1.0
        try:
            if "scale" in self.args:
                self.scale = float(self.args["scale"])
            if not self.scale in [0.25, 0.5, 1.0, 2.0, 4.0]:
                self.log.warn("Invalid scale value, must be 0.25, 0.5, 1.0, 2.0 or 4.0: defaulting to 1.0")
                raise ValueError
        except ValueError:
            self.scale = 1.0

        self.exp = 1
        try:
            if "exp" in self.args:
                self.exp = int(self.args["exp"])
            if self.exp < 1:
                self.log.warn("Invalid exp value: defaulting to 1")
                raise ValueError
        except ValueError:
            self.exp = 1

        gpu = torch.cuda.is_available()

        self.deviceDType = torch.float32
        if gpu and self.fp16:
            if self.height >= 2048 or self.width >= 2048:
                self.fp16 = False
                self.log.warn("Disabling FP16 for resolution >= 2048x2048")
            else:
                self.deviceDType = torch.float16
        else:
            self.fp16 = False

        if(self.fp16):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        self.device = torch.device("cuda" if gpu else "cpu")

        # Load the model
        self.model = Model()
        self.model.load_model(current_dir+os.path.sep+'train_log', -1)

        torch.set_grad_enabled(False)
        self.model.eval()
        self.model.device()
        self.tmpFrame = None

        self.reader = args['reader']
        self.writer = args['writer']
        if self.fps > 0:
            args['writer'].fps *= (1 << self.exp)
        self.outBatchSize = self.inBatchSize * (1 << self.exp)

        tmp = max(32, int(32 / self.scale))
        ph = ((self.height - 1) // tmp + 1) * tmp
        pw = ((self.width - 1) // tmp + 1) * tmp
        self.padding = (0, pw - self.width, 0, ph - self.height)

    def make_inference(self, I0, I1, n):
        middle = self.model.inference(I0, I1, self.scale)
        if n == 1:
            return [middle]
        first_half = self.make_inference(I0, middle, n=n//2)
        second_half = self.make_inference(middle, I1, n=n//2)
        if n%2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

    def pad_image(self, img):
        if(self.fp16):
            return F.pad(img, self.padding).half()
        else:
            return F.pad(img, self.padding)

    def process(self, image, frameNumber):
        assert(image.shape[1:] == (3, self.height, self.width))
        out = []
        while True:
            if self.tmpFrame is not None:
                frame = self.tmpFrame 
                self.tmpFrame  = None
            else:
                frame = image

            if self.lastframe is None:
                self.lastframe = frame[:]
                self.I1 = torch.from_numpy(self.lastframe).to(self.device, dtype=self.deviceDType, non_blocking=True)
                self.I1 = self.pad_image(self.I1)
                return frame

            self.I0 = self.I1
            self.I1 = torch.from_numpy(frame).to(self.device, dtype=self.deviceDType, non_blocking=True)
            self.I1 = self.pad_image(self.I1)
            I0_small = F.interpolate(self.I0, (32, 32), mode='bilinear', align_corners=False)
            I1_small = F.interpolate(self.I1, (32, 32), mode='bilinear', align_corners=False)
            ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

            if self.useHighSsim and ssim > 0.996 and len(out) < 16:
                try:
                    frame = self.reader.read() # read a new frame
                    self.tmpFrame = frame
                except StopIteration:
                    frame = self.lastframe
                    self.tmpFrame = None
                self.I1 = torch.from_numpy(frame).to(self.device, dtype=self.deviceDType, non_blocking=True)
                self.I1 = self.pad_image(self.I1)
                self.I1 = self.model.inference(self.I0, self.I1, self.scale)
                I1_small = F.interpolate(self.I1, (32, 32), mode='bilinear', align_corners=False)
                ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
                #frame = (self.I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:self.height, :self.width]
                frame = self.I1.to(device="cpu", dtype=torch.float32).numpy()[:, :, :self.height, :self.width]
            
            if ssim < 0.2:
                output = []
                for i in range((2 ** self.exp) - 1):
                    output.append(self.I0)
                '''
                output = []
                step = 1 / (2 ** self.exp)
                alpha = 0
                for i in range((2 ** self.exp) - 1):
                    alpha += step
                    beta = 1-alpha
                    output.append(torch.from_numpy(np.transpose((cv2.addWeighted(frame[:, :, ::-1], alpha, lastframe[:, :, ::-1], beta, 0)[:, :, ::-1].copy()), (2,0,1))).to(self.device, non_blocking=True).unsqueeze(0).float() / 255.)
                '''
            else:
                output = self.make_inference(self.I0, self.I1, 2**self.exp-1) if self.exp else []

            #out.append(self.lastframe[0, :])
            out.append(self.lastframe)
            for i, mid in enumerate(output):
                #mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0)))
                #mid = mid.to(device="cpu", dtype=torch.float32).numpy()
                #out.append(mid[0, :, :self.height, :self.width])
                out.append(mid.to(device="cpu", dtype=torch.float32).numpy()[:, :, :self.height, :self.width])

            self.lastframe = frame
            if self.tmpFrame is None:
                while len(out) > 1:
                    self.writer.write(out.pop(0))
                return out.pop(0)

filters["rife"] = RIFEFilter