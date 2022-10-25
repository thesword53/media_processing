from filter import Filter, filters
from wrappers.rife.model.pytorch_msssim import ssim_matlab
import torch
import torch.nn.functional as nnf
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

class DifFilter(Filter):
    def __init__(self, args, filter):
        self.args = argsToDict(args["user"])
        self.reader = args["reader"]
        self.reader.dtype = np.float32
        self.filter = filter
        self.doubled = True
        self.reader.batchSize = 4 * self.filter.inBatchSize
        self.inBatchSize = 4 * self.filter.inBatchSize
        self.outBatchSize = 2 * self.filter.outBatchSize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #try:
        #    if "filter" in self.args:
        #        self.filter = self.args["filter"]
        #except KeyError:
        #    raise DifFilterArgumentError(f"\"filter\" {self.args['filter']} not found")

    def process(self, image, frameNumber):
        if image.shape[2] % 4 != 0:
            return
        img = torch.from_numpy(image[:2]).to(self.device, dtype=torch.float32, non_blocking=True)
        img1 = img[:, :, 0::2]
        img2 = img[:, :, 1::2]
        img1_small = nnf.interpolate(img1, (256, 256), mode='bilinear', align_corners=False)
        img2_small = nnf.interpolate(img2, (256, 256), mode='bilinear', align_corners=False)
        if ssim_matlab(img1_small, img2_small) < 0.98:
            out1 = self.filter.process(image[2:3], frameNumber)
            out2 = self.filter.process(image[3:4], frameNumber)
        else:
            out1 = out2 = self.filter.process(image[:1], frameNumber)
        out = np.zeros((2,) + out1.shape[1:], dtype=np.float32)
        out[0:1] = out1
        out[1:2] = out2
        return out

#filters["dif"] = DifFilter