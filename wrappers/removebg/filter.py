from filter import *
import flogging

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np

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

preprocess = transforms.Compose([
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class RemoveBGFilter(Filter):
    def __init__(self, args):
        super().__init__(args)
        self.log = flogging.FilterLogging(args["loglevel"], "RemoveBG")
        self.args = argsToDict(args["user"])
        self.noise = -1
        self.width = self.inputWidth = args["reader"].width
        self.height = self.inputHeight = args["reader"].height
        self.reader = args["reader"]
        self.reader.dtype = np.float32
        self.reader.batchSize = 1
        self.reader._4D = True
        self.numFrames = self.reader.numFrames
        self.fps = self.reader.fps

        self.fp16 = True

        gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if gpu else "cpu")

        if gpu and self.fp16:
            self.deviceDType = torch.float16
        else:
            self.deviceDType = torch.float32
            self.fp16 = False

        if(self.fp16):
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        self.log.print("Using {:s} with {:s}".format("GPU" if gpu else "CPU", "FP16" if self.fp16 else "FP32"))

        self.model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True)
        self.model = self.model.to(device=self.device)
        self.model.eval()
        self.log.print("Model Loaded")
        self.people_class = 15
        self.blur = torch.from_numpy(np.array([[[[1.0, 2.0, 1.0],[2.0, 4.0, 2.0],[1.0, 2.0, 1.0]]]]) / 16.0).to(self.device, dtype=self.deviceDType, non_blocking=True)

    def process(self, frame, frameNumber):
        #frame_data = torch.FloatTensor( img ) / 255.0

        #input_tensor = preprocess(frame_data.permute(2, 0, 1))
        #input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
        input_batch = preprocess(torch.from_numpy(frame).to(self.device, dtype=self.deviceDType, non_blocking=True))

        # move the input and model to GPU for speed if available
        #if torch.cuda.is_available():
        #    input_batch = input_batch.to('cuda')


        with torch.no_grad():
            output = self.model(input_batch)['out'][0]

        segmentation = output.argmax(0)

        bgOut = output[0:1][:][:]
        a = (1.0 - F.relu(torch.tanh(bgOut * 0.30 - 1.0))).pow(0.5) * 2.0

        if self.fp16:
            people = segmentation.eq( torch.ones_like(segmentation).long().fill_(self.people_class) ).half()
        else:
            people = segmentation.eq( torch.ones_like(segmentation).long().fill_(self.people_class) ).float()

        people.unsqueeze_(0).unsqueeze_(0)
        
        for i in range(3):
            people = F.conv2d(people, self.blur, stride=1, padding=1)

        # combined_mask = F.hardtanh(a * b)
        combined_mask = F.relu(F.hardtanh(a * (people.squeeze().pow(1.5)) ))
        #combined_mask = combined_mask.expand(1, 3, -1, -1)

        #res = (combined_mask * 255.0).cpu().squeeze().byte().permute(1, 2, 0).numpy()

        mask = torch.clamp(combined_mask, 0.0, 1.0).to(device="cpu", dtype=torch.float32).numpy()
        out = np.ones((1, 4, frame.shape[2], frame.shape[3]), dtype=np.float32)
        out[0,:3,:,:] = frame
        out[0,3,:,:] = mask
        return out

    def read(self):
        return self.process(self.reader.read(), 0)

filters["RemoveBG"] = RemoveBGFilter
