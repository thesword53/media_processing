#! /usr/bin/python
# -*- coding: utf8 -*-

from ffmpeg_tools import FFmpegReader, FFmpegWriter
import argparse
import numpy as np
from converter import convertFrames
import sys
import flogging
import wrappers.filterlist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('-m2', dest='model2', action='store_true', help='Use model 2')

    parser.add_argument('output', type=str, help='Output file')
    parser.add_argument('-i', dest='input', default='', type=str, help='Input file')
    parser.add_argument('-ia', dest='input_args', default='', help='Input ffmpeg arguments: "arg1;arg2;..."')
    parser.add_argument('-oa', dest='output_args', default='', help='Output ffmpeg arguments: "arg1;arg2;..."')
    parser.add_argument('-wa', dest='wrapper_args', default='', help='Wrapper arguments: "arg1;arg2;..."')
    parser.add_argument('-f', '-w', dest='filter', default='', help='Filter')
    parser.add_argument('-fps', dest='fps', default='', type=str, help='Force input fps')
    parser.add_argument('-v', '--debug', dest='debug', action='store_true', help='Enable debug logs')
    parser.add_argument('--loglevel', dest='loglevel', default='', type=str, help='Enable more verbosity')
    parser.add_argument('--use-cuvid', dest='use_cuvid', action='store_true', help='Use CUVID for videos deinterlacing')

    args = parser.parse_args()
    converterArgs = {}

    if args.loglevel == "fatal":
        args.loglevel = flogging.LOGLEVEL_FATAL
    elif args.loglevel in ["err", "error"]:
        args.loglevel = flogging.LOGLEVEL_ERROR
    elif args.loglevel in ["warn", "warning"]:
        args.loglevel = flogging.LOGLEVEL_WARNING
    elif args.loglevel in ["", "normal", "verbose"]:
        args.loglevel = flogging.LOGLEVEL_VERBOSE
    elif args.loglevel in ["dbg", "debug"]:
        args.loglevel = flogging.LOGLEVEL_DEBUG
    else:
        print("\033[91mInvalid log level {:s}\033[0m".format(args.loglevel))
        sys.exit(1)

    if args.debug:
        args.loglevel = flogging.LOGLEVEL_DEBUG

    logger = flogging.FilterLogging(args.loglevel)

    loader = FFmpegReader(args.input, args.loglevel, args.input_args, useCuvid=args.use_cuvid)

    if len(args.fps) > 0:
        try:
            loader.fps = float(args.fps)
            if loader.fps < 1.0:
                raise ValueError("fps < 0")
        except ValueError:
            print("Fps must be a float > 0")
            sys.exit(1)

    writer = FFmpegWriter(args.output, args.loglevel, args.output_args, np.uint8, loader.fps)

    converterArgs["metadata"] = loader.jsonDict
    converterArgs["user"] = args.wrapper_args
    converterArgs["filter"] = args.filter
    converterArgs["reader"] = loader
    converterArgs["writer"] = writer
    converterArgs["loglevel"] = args.loglevel
    converterArgs["logger"] = logger

    try:
        convertFrames(loader, writer, converterArgs)
    except KeyboardInterrupt:
        logger.print("Interrupted")
        loader.close()
        writer.close()
        sys.exit(1)
    except:
        loader.close()
        writer.close()
        raise

    loader.close()
    writer.close()