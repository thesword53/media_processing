import copy
import flogging
from filter import *
from wrappers.Waifu2x.filter import Waifu2xFilter
from wrappers.rife.filter import RIFEFilter

class Waifu2xRIFEFilter(Filter):
    def __init__(self, args):
        self.log = flogging.FilterLogging(args["loglevel"], "Waifu2xRIFE")
        wrapperArgList = args["user"].split(":")
        if len(wrapperArgList) != 2:
            raise Waifu2xRIFEFilterArgumentError("Both wrapper args must be separated by \":\"")
        wargs = copy.copy(args)
        wargs["user"] = wrapperArgList[0]
        self.upscaler = Waifu2xFilter(wargs, TILE_SIZE=512)
        self.inBatchSize = self.upscaler.inBatchSize
        rargs = copy.copy(args)
        rargs["user"] = wrapperArgList[1]
        rargs["reader"] = self.upscaler
        self.rife = RIFEFilter(rargs, fp16=(self.upscaler.width < 2048 and self.upscaler.height < 2048))
        self.outBatchSize = self.rife.outBatchSize

    def process(self, image, frameNumber):
        frame = self.upscaler.process(image, frameNumber)
        return self.rife.process(frame, frameNumber)

class Waifu2xRIFEFilterArgumentError(FilterArgumentError):
    def __init__(self, text):
        self.text = text
        self.filter = "Waifu2xRIFE"

    def __str__(self):
        return self.text

filters["waifu2xrife"] = Waifu2xRIFEFilter
