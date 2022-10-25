import os
import sys

class Filter(object):
    def __init__(self, args):
        self._getVideoInfo(args)

    def _getVideoInfo(self, args):
        reader = args['reader']
        self.width = reader.width
        self.height = reader.height
        self.numFrames = reader.numFrames
        self.fps = reader.fps
        self.inBatchSize = 1 #-1 -> undef
        self.outBatchSize = 1 #-1 -> undef

    def process(self, image, frameNumber):
        return image

filters = {"": Filter, "identity": Filter}

class FilterArgumentError(Exception):
    def __init__(self, text):
        self.text = text
        self.filter = "Identity"

    def __str__(self):
        return self.text
