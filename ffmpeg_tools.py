import subprocess as sp
import numpy as np
import json
import flogging
import time
import threading
import queue

dtypeToPixFmt = {np.uint8: "rgb24", np.uint16: "rgb48le", np.float32: "gbrpf32le", np.dtype('uint16'): "rgb48le", np.dtype('uint8'): "rgb24", np.dtype('float32'): "gbrpf32le"}
dtypeToAlphaPixFmt = {np.uint8: "rgba", np.uint16: "rgba64le", np.float32: "gbrapf32le", np.dtype('uint16'): "rgba64le", np.dtype('uint8'): "rgba", np.dtype('float32'): "gbrapf32le"}

def argsToList(argsStr):
    argList = []
    if len(argsStr) > 0:
        argsStrList = argsStr.split(';')
        for arg in argsStrList:
            elts = arg.split('=', 1)
            argList.append('-'+elts[0])
            if len(elts) >= 2:
                argList.append(elts[1])
    return argList

class FFmpegReader(threading.Thread):
    def __init__(self, input, loglevel, args, useCuvid=False):
        super().__init__()
        self.queue = queue.Queue(64)
        self.log = flogging.FilterLogging(loglevel, "FFmpegReader")
        procArgs = ["ffprobe", "-v", "error", "-show_streams", "-print_format", "json", input]
        proc2 = sp.Popen(procArgs, stdout=sp.PIPE, stderr=sp.DEVNULL)
        output, err = proc2.communicate()
        retcode = proc2.poll()
        if retcode != 0:
            error = sp.CalledProcessError(retcode, procArgs)
            error.output = output
            raise error

        self._input = input
        self.jsonDict = json.loads(output)

        self._getVideoInfo(self.jsonDict)
        self.readFrames = 0
        self._probeFilters(args)
        self.dtype = np.uint8
        self.batchSize = 1
        self._4D = False
        self.args = args
        self.device = None
        self.eof = False
        self.useDifFilter = False 

        #import json
        #print json.dumps(d, indent = 4)
        #exit(0)
        self.douleFrame = self.fps > 30.0
        self.filterArgs = []
        self.decoderArgs = []
        if useCuvid and self.codec in ["mjpeg", "mpeg1video", "mpeg2video", "mpeg4", "vc1", "h264", "hevc", "vp8", "vp9", "av1"]:
            self.numFrames *= 2
            if not self.douleFrame:
                self.fps *= 2.0
            
            self.decoderArgs = ["-c:v", f"{self.codec.replace('video', '')}_cuvid"] + ([] if self.douleFrame else ["-r", f"{self.fps}"]) + ["-deint", "adaptive"]
        else:
            if useCuvid:
                self.log.warn(f"Unable to use cuvid with \"{self.codec}\" codec")
            if self.fieldOrder == "tt" or self.fieldOrder == "tb":
                if not self.douleFrame:
                    self.fps *= 2.0
                    self.numFrames *= 2
                self.filterArgs = ["-filter_complex", f"[0:v] fps={self.fps} [x];[0:v] yadif=1,fps={self.fps} [y]; [x][y] framepack=format=frameseq [z]; [z] shuffleframes=0 2 1 3"]
                self.useDifFilter = True
            elif self.fieldOrder == "bb" or self.fieldOrder == "bt":
                if not self.douleFrame:
                    self.fps *= 2.0
                    self.numFrames *= 2
                    self.useDifFilter = True
                self.filterArgs = ["-filter_complex", f"[0:v] fps={self.fps} [x];[0:v] yadif=1,fps={self.fps} [y]; [x][y] framepack=format=frameseq [z]; [z] shuffleframes=0 2 1 3"]

        if len(args) > 0:
            for arg in args.split(';'):
                if "fps=" in arg:
                    try:
                        fps = int(arg.split("fps=")[1].split(",")[0])
                        self.log.debug("setting fps to {:s}", fps)
                        self.fps = fps
                    except ValueError:
                        pass
                    break
        self.proc = None
        #retcode = self.proc.poll()
        #if retcode != 0:
        #    error = sp.CalledProcessError(retcode, procArgs)
        #    raise error

    def _probeFilters(self, args):
        try:
            for arg in args:
                if len(args) > 3 and args[:3] == "vf=":
                    for a in args[3:].split(","):
                        k, v = a.split("=")
                        if k in ["yadif", "yadif_cuda"]:
                            if v in ["1", "3", "send_field", "send_field_nospatial"]:
                                self.fps *= 2.0
                                self.numFrames *= 2
                                self.log.debug("Doubling fps and frame number")
                        if k == "fps":
                            if "/" in v:
                                vs = v.split("/")
                                newFps = int(vs[0]) / int(vs[1])
                            elif v == "source_fps":
                                newFps = self.fps
                            elif v == "ntsc":
                                newFps = 30000 / 1001
                            elif v == "pal":
                                newFps = 25.0
                            elif v == "film":
                                newFps = 24.0
                            elif v == "ntsc_film":
                                newFps = 24000 / 1001
                            else:
                                newFps = float(v)
                            if newFps != self.fps:
                                self.numFrames = int(self.numFrames * (newFps/self.fps))
                            assert(newFps > 0.0)
                            self.fps = newFps
                            self.log.debug("Setting fps to {:f}", self.fps)
                        if k == "crop":
                            in_w = self.width
                            int_h = self.height
                            for i, v in enumerate(v.split(":")):
                                if (len(v) > 2 and v[:2] == "w=") or (not "=" in v and i == 0):
                                    val = int(v[2:]) if "=" in v else int(v)
                                    assert(val > 0)
                                    self.width = min(self.width, val)
                                if (len(v) > 2 and v[:2] == "h=") or (not "=" in v and i == 1):
                                    val = int(v[2:]) if "=" in v else int(v)
                                    assert(val > 0)
                                    self.height = min(self.height, val)
                            self.log.debug("Setting scale to {:d}x{:d}", self.width, self.height)
                break
        except ValueError:
            self.log.warn("probeFilters: ValueError")
        except IndexError:
            self.log.warn("probeFilters: IndexError")
        except AssertionError:
            self.log.warn("AssertionError: IndexError")
        self.log.debug("Filtered video: {:d}x{:d}, {:f} FPS, {:d} frames", self.width, self.height, self.fps, self.numFrames)
    def _probCountFrames(self):
        # open process, grabbing number of frames using ffprobe
        probecmd = ["ffprobe"] + ["-v", "error", "-count_frames", "-select_streams", "v:0",
                                                  "-show_entries", "stream=nb_read_frames", "-of",
                                                  "default=nokey=1:noprint_wrappers=1", self._input]
        proc = sp.Popen(probecmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)
        nbFrames = np.int(proc.stdout.read().decode('utf8').split('\n')[0])
        self.log.debug("Probed {:d} frames", nbFrames)
        return nbFrames

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
        self.depth = 3
        self.numFrames = -1
        if "nb_frames" in videoStream:
            self.numFrames = int(videoStream["nb_frames"])

        avgFps = videoStream["avg_frame_rate"]
        if avgFps == "0/0":
            self.fps = float(videoStream["duration"]) / self.numFrames
        else:
            avgFpsList = avgFps.split("/")
            self.fps = int(avgFpsList[0]) / int(avgFpsList[1])

        if self.numFrames == -1:
            self.numFrames = self._probCountFrames()
        self.fieldOrder = videoStream["field_order"] if "field_order" in videoStream else "unknown"
        self.codec = videoStream["codec_name"] if "codec_name" in videoStream else "unknown"
        self.log.debug("Video: {:d}x{:d}, {:f} FPS, {:d} frames", self.width, self.height, self.fps, self.numFrames)

    def __iter__(self):
        return self

    def __next__(self):
        return self.read()

    def next(self):
        return self.read()

    def read(self):
        if self.proc is None:
            procArgs = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "warning"] + self.decoderArgs + ["-i", self._input] + self.filterArgs + ["-c:v", "rawvideo", "-pix_fmt", dtypeToPixFmt[self.dtype], "-f", "image2pipe"] + argsToList(self.args) + ["-"]
            self.proc = sp.Popen(procArgs, stdin=sp.PIPE, stdout=sp.PIPE, stderr=None)
            self.start()

        if self.eof:
            raise StopIteration()

        arr = self.queue.get()
        if arr is None:
            self.eof = True
            raise StopIteration()

        self.readFrames += arr.shape[0] if len(arr.shape) == 4 else 1
        return arr

    def run(self):
        while True:
            if self.proc is None:
                self.queue.put(None)
                break
            imageSize = self.width*self.height*self.depth*np.dtype(self.dtype).itemsize
            buff = self.proc.stdout.read(self.batchSize*imageSize)
            if len(buff) < imageSize:
                self.queue.put(None)
                break

            batchSize = self.batchSize
            if self.batchSize > 1 or self._4D:
                batchSize = len(buff) // imageSize
                if self.dtype == np.float32:
                    arr = np.frombuffer(buff, dtype=self.dtype).reshape((batchSize, self.depth, self.height, self.width))[:, [2,0,1]]
                else:
                    arr = np.frombuffer(buff, dtype=self.dtype).reshape((batchSize, self.height, self.width, self.depth))
            else:
                if self.dtype == np.float32:
                    arr = np.frombuffer(buff, dtype=self.dtype).reshape((self.depth, self.height, self.width))[[2,0,1]]
                else:
                    arr = np.frombuffer(buff, dtype=self.dtype).reshape((self.height, self.width, self.depth))
            while True:
                try:
                    self.queue.put(arr, timeout=5)
                    break
                except queue.Full:
                    if self.proc is None:
                        return
                    continue

    def close(self):
        if self.proc is None:  # pragma: no cover
            return  # no process
        if self.proc.poll() is not None:
            return  # process already dead
        if self.proc.stdin:
            self.proc.stdin.close()
        if self.proc.stdout:
            self.proc.stdout.close()
        if self.proc.stderr:
            self.proc.stderr.close()
        self.proc.wait()
        self.proc = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class FFmpegWriter(threading.Thread):
    def __init__(self, output, loglevel, args, dtype, fps=25):
        super().__init__()
        self.queue = queue.Queue(64)
        self.log = flogging.FilterLogging(loglevel, "FFmpegWriter")
        self.width = -1
        self.height = -1
        self.depth = -1
        self.dtype = dtype
        self.initialized = False
        self.proc = None
        self.fps = fps
        self.args = args
        self.output = output
        self.device = None

    def write(self, arrayData):
        if not self.initialized:
            if len(arrayData.shape) == 4:
                height, width, depth = arrayData.shape[1:]
            else:
                height, width, depth = arrayData.shape
            if arrayData.dtype == np.dtype('float32'): # Planar
                depth, height, width = height, width, depth
            self.height = height
            self.width = width
            self.depth = depth
            self.dtype = arrayData.dtype
            getFmt = dtypeToAlphaPixFmt if self.depth == 4 else dtypeToPixFmt
            self.log.debug("Initializing writer: frame info: {:d}x{:d}, {:f} FPS, format: {:s}", self.width, self.height, self.fps, getFmt[self.dtype])
            procArgs = ["ffmpeg", "-hide_banner", "-nostats", "-loglevel", "warning", "-y", "-pix_fmt", getFmt[self.dtype], "-f", "rawvideo", "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-"] + argsToList(self.args) + [self.output]
            self.proc = sp.Popen(procArgs, stdin=sp.PIPE, stdout=sp.DEVNULL, stderr=sp.STDOUT)
            self.initialized = True
            self.start()
        
        self.queue.put(arrayData)

    def run(self):
        while True:
            if self.proc is None:
                break
            try:
                arrayData = self.queue.get(timeout=5)
            except queue.Empty:
                if self.proc is None:
                    break
                continue
            if arrayData is None:
                break
            if len(arrayData.shape) == 4:
                height, width, depth = arrayData.shape[1:]
            else:
                height, width, depth = arrayData.shape
            if arrayData.dtype == np.dtype('float32'): # Planar
                depth, height, width = height, width, depth
                transpose = (1,2,0,3) if depth == 4 else (1,2,0)
                if len(arrayData.shape) == 4:
                    arrayData = arrayData[:, transpose]
                else:
                    arrayData = arrayData[transpose]

            assert(self.height == height)
            assert(self.width == width)
            assert(self.depth == depth)
            assert(self.dtype == arrayData.dtype)
            assert(self.depth in (3,4))
            
            self.proc.stdin.write(arrayData.tobytes())

    def close(self):
        if self.proc is None:  # pragma: no cover
            return  # no process
        if self.proc.poll() is not None:
            return  # process already dead
        self.queue.put(None)
        self.join()
        if self.proc.stdin:
            self.proc.stdin.close()
        if self.proc.stdout:
            self.proc.stdout.close()
        if self.proc.stderr:
            self.proc.stderr.close()
        self.proc.wait()
        self.proc = None
        self.initialized = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
