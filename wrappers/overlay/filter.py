import json
import datetime
from filter import Filter, filters
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import flogging

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

def drawOverlay(info, image, x, y, size):
    # Availability is platform dependent
    font = 'arial'
    
    # Create font
    pil_font = ImageFont.truetype(font + ".ttf", size=size,
                                  encoding="unic")

    # create a blank canvas with extra space between lines
    canvas = Image.fromarray(image)

    # draw the text onto the canvas
    draw = ImageDraw.Draw(canvas)
    offset = [x, y]


    white = "#000000"
    draw.text(offset, "Speed: {:.1f} m/s [{:.1f} km/h]".format(info["speed"], info["speed"] * 3.6), font=pil_font, fill=white)
    offset[1] += size * 1.5
    draw.text(offset, "Altitude: {:d} m".format(int(info["altitude"])), font=pil_font, fill=white)
    offset[1] += size * 1.5
    draw.text(offset, "Angle: {:d}°".format(int(info["angle"])), font=pil_font, fill=white)

    # Convert the canvas into an array with values in [0, 1]
    return np.asarray(canvas)

class gpsOverlayFilter(Filter):
    def __init__(self, args):
        self.log = flogging.FilterLogging(args["loglevel"], "gpsOverlay")
        self.args=args
        argDict = argsToDict(args["user"])
        if argDict == None or not "path" in argDict.keys():
            self.log.error("Invalid arguments: args: path=<path>[;dt=<deltatime>]")
            raise ValueError("GPS overlay: args: path=<path>[;dt=<deltatime>]")

        if "dt" in argDict.keys():
            self.startTime = float(argDict["dt"])
        else:
            self.startTime = 0.0
        with open(argDict["path"], "r") as f:
            self.gpsList = json.loads(f.read())
        self.metaData = args["metadata"]
        self._getVideoInfo(self.metaData)
        self.id = 0
        self.gpsCount = len(self.gpsList)
        for i, g in reversed(list(enumerate(self.gpsList))):
            cTime = float(g["date"])
            if cTime >= self.startTime:
                self.id = i

    def _getVideoInfo(self, jsonDict):
        streams = jsonDict["streams"]
        videoStream = None
        for s in streams:
            if s["codec_type"] == "video":
                videoStream = s
        if videoStream == None:
            raise ValueError("No video stream found")
        self.width = videoStream["width"]
        self.height = videoStream["height"]
        self.numFrames = int(videoStream["nb_frames"])

        avgFps = videoStream["avg_frame_rate"]
        if avgFps == "0/0":
            self.fps = float(videoStream["duration"]) / self.numFrames
        else:
            avgFpsList = avgFps.split("/")
            self.fps = int(avgFpsList[0]) / int(avgFpsList[1])
        self.startTime += datetime.datetime.fromisoformat(videoStream["tags"]["creation_time"][:-1]).timestamp()

    def getInfo(self, frameNumber):
        currTime = self.startTime + frameNumber / self.fps
        while True:
            if self.id < self.gpsCount - 1:
                if currTime >= float(self.gpsList[self.id+1]["date"]):
                    self.id += 1
                else:
                    break
            else:
                break
        gps = self.gpsList[self.id]
        info = {}
        info["time"] = float(gps["date"])
        info["latitude"] = gps["coords"]["latitude"]
        info["longitude"] = gps["coords"]["longitude"]
        info["altitude"] = gps["coords"]["altitude"]
        info["accuracy"] = gps["coords"]["accuracy"]
        info["angle"] = gps["coords"]["bearing"]
        info["speed"] = gps["coords"]["speed"]
        return info

    def process(self, image, frameNumber):
        info = self.getInfo(frameNumber)

        return drawOverlay(info, image, 50, 50, 20)

filters["gps"] = gpsOverlayFilter
filters["gps_overlay"] = gpsOverlayFilter