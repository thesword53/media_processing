from filter import *
from Models import *
from utils.prepare_images import *
from PIL import Image
import numpy as np
import flogging
import torch.nn.functional as nnf

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

class Waifu2xFilter(Filter):
    def __init__(self, args, TILE_SIZE=1024):
        super().__init__(args)
        self.log = flogging.FilterLogging(args["loglevel"], "Waifu2x")
        self.args = argsToDict(args["user"])
        self.noise = -1
        self.inputWidth = args["reader"].width
        self.inputHeight = args["reader"].height
        self.reader = args["reader"]
        self.reader.dtype = np.float32
        self.reader.batchSize = 1
        self.reader._4D = True
        self.numFrames = self.reader.numFrames
        self.fps = self.reader.fps
        self.boarderPadSize = 3

        #if True:
        #    self.reader.batchSize = 16
        #    self.boarderPadSize = 8

        try:
            if "denoise" in self.args:
                self.noise = int(self.args["denoise"])
            if not self.noise in [-1, 0, 1, 2, 3]:
                raise Waifu2xFilterArgumentError("\"denoise\" must be -1 (no denoising), 0, 1, 2 or 3")
        except ValueError:
            raise Waifu2xFilterArgumentError("\"denoise\" must be -1 (no denoising), 0, 1, 2 or 3")

        self.exp = 0
        self.resample = False

        width = 0
        height = 0
        try:
            if "width" in self.args:
                width = int(self.args["width"])
                height = int(round(width * (self.inputHeight / self.inputWidth)))
        except ValueError:
            raise Waifu2xFilterArgumentError("\"width\" must be an integer")

        try:
            if "height" in self.args:
                height = int(self.args["height"])
                if width == 0:
                    width = int(round(height * (self.inputWidth / self.inputHeight)))
        except ValueError:
            raise Waifu2xFilterArgumentError("\"height\" must be an integer")

        try:
            if "scale" in self.args:
                if "x" in self.args["scale"]:
                    width, height = self.args["scale"].split("x")
                    width = int(width)
                    height = int(height)
                else:
                    raise ValueError
        except ValueError:
            raise Waifu2xFilterArgumentError("\"scale\" must be <integer>x<integer>")

        self.resampleLast = True
        if width > 0:
            if width <= self.inputWidth:
                raise Waifu2xFilterArgumentError("\"width\" must be greater than input width")
            if height <= self.inputHeight:
                raise Waifu2xFilterArgumentError("\"height\" must be greater than input height")
            self.resample = np.log2(width/self.inputWidth) != self.exp or np.log2(height/self.inputHeight) != self.exp
            self.exp = int(np.ceil(max(np.log2(width/self.inputWidth), np.log2(height/self.inputHeight))))
            self.resampleLast = self.exp == 1 or ((width // (self.exp-1)) * (self.exp-1)) != width

        try:
            if "exp" in self.args:
                if self.exp != 0:
                    raise Waifu2xFilterArgumentError("\"exp\" must no be set if video size is set")
                self.exp = int(self.args["exp"])
            if self.exp < 1:
                raise Waifu2xFilterArgumentError("\"exp\" must be > 1")
        except ValueError:
            raise Waifu2xFilterArgumentError("\"exp\" must be an integer > 1")

        self.width = width if width > 0 else self.inputWidth << self.exp
        self.height = height if height > 0 else self.inputHeight << self.exp

        self.fp16 = True

        gpu = torch.cuda.is_available()
        self.device = torch.device("cuda" if gpu else "cpu")

        if gpu and self.fp16:
            self.deviceDType = torch.float16
        else:
            self.deviceDType = torch.float32
            self.fp16 = False

        self.log.print("Using {:s} with {:s}".format("GPU" if gpu else "CPU", "FP16" if self.fp16 else "FP32"))

        model = "Upconv_7/anime"
        if "model" in self.args:
            model = self.args["model"]

        self.log.debug("Using noise level {:d}", self.noise)
        self.log.debug("Using model \"{:s}\"", model)

        self.model2 = UpConv_7()
        self.model2.load_pre_train_weights("wrappers"+os.path.sep+"Waifu2x"+os.path.sep+"model_check_points/{:s}/scale2.0x_model.json".format(model))
        self.model2 = self.model2.to(device=self.device)
        if self.fp16:
            self.model2 = network_to_half(self.model2)
        if self.noise in [0, 1, 2, 3]:
            self.model = UpConv_7()
            self.model.load_pre_train_weights("wrappers"+os.path.sep+"Waifu2x"+os.path.sep+"model_check_points/{:s}/noise{:d}_scale2.0x_model.json".format(model, self.noise))
            self.model = self.model.to(device=self.device)
            if self.fp16:
                self.model = network_to_half(self.model)
        else:
            self.model = self.model2
        self.scaleFactor = 2 ** self.exp
        self.tileSize = TILE_SIZE // (self.scaleFactor//2) // int(sqrt(self.reader.batchSize))
        self.log.debug("Scale factor is {:d} with {:d}x{:d} tiles", self.scaleFactor, self.tileSize, self.tileSize)
        self.iterationCount = self.exp

    def process(self, image, frameNumber):
        if image.shape[1] > self.tileSize or image.shape[2] > self.tileSize or image.shape[3] > self.tileSize:
            imgSplitter = ImageSplitter(seg_size=self.tileSize, scale_factor=self.scaleFactor, boarder_pad_size=self.boarderPadSize)
            if self.resample and not self.resampleLast:
                imgSplitter = ImageSplitter(seg_size=self.tileSize, scale_factor=2, boarder_pad_size=self.boarderPadSize)
            imgPatches = imgSplitter.split_img_tensor(image, scale_method=None, img_pad=0)
            imgPatches = [p.to(device=self.device, dtype=self.deviceDType, non_blocking=True) for p in imgPatches]

            with torch.no_grad():
                out = [self.model(i) for i in imgPatches]
                if self.resample and not self.resampleLast:
                    out = imgSplitter.merge_img_tensor(out, batch_size=image.shape[0])
                    imgSplitter = ImageSplitter(seg_size=self.tileSize, scale_factor=self.scaleFactor>>1, boarder_pad_size=self.boarderPadSize)
                    out = nnf.interpolate(out, size=(self.height>>(self.exp-1), self.width>>(self.exp-1)), mode='bicubic', align_corners=False)
                    out = imgSplitter.split_img_tensor(out, scale_method=None, img_pad=0)
                    out = [p.to(device=self.device, dtype=self.deviceDType, non_blocking=True) for p in out]
                for j in range(self.iterationCount-1):
                    out = [self.model2(i) for i in out]

                out = imgSplitter.merge_img_tensor(out, batch_size=image.shape[0])
                if self.resample and self.resampleLast:
                    out = nnf.interpolate(out, size=(self.height, self.width), mode='bicubic', align_corners=False)

            img = torch.clamp(out, 0.0, 1.0).to(device="cpu", dtype=torch.float32).numpy()
        else:
            with torch.no_grad():
                out = self.model(torch.from_numpy(image).to(device=self.device, dtype=self.deviceDType, non_blocking=True))
                if self.resample and not self.resampleLast:
                    out = nnf.interpolate(out, size=(self.height>>(self.exp-1) , self.width>>(self.exp-1)), mode='bicubic', align_corners=False)
                for j in range(self.iterationCount-1):
                    out = [self.model2(i) for i in out]
                if self.resample and self.resampleLast:
                    out = nnf.interpolate(out, size=(self.height, self.width), mode='bicubic', align_corners=False)
                out = torch.clamp(out, 0.0, 1.0)
                img = out.to(device="cpu", dtype=torch.float32).numpy()

        return img
        #return img.astype(np.uint16).transpose(0, 2, 3, 1)

    def read(self):
        return self.process(self.reader.read(), 0)

filters["waifu2x"] = Waifu2xFilter

class Waifu2xFilterArgumentError(FilterArgumentError):
    def __init__(self, text):
        self.text = text
        self.filter = "Waifu2x"

    def __str__(self):
        return self.text