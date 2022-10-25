#! /usr/bin/python
# -*- coding: utf8 -*-

import sys, os
sys.path.append(os.path.join(os.getcwd(), 'wrappers'+os.path.sep+'Waifu2x'))
sys.path.append(os.path.join(os.getcwd(), 'wrappers'+os.path.sep+'rife'))
import numpy as np
from filter import filters, FilterArgumentError
from wrappers.dif.filter import DifFilter
#import time
from tqdm import tqdm
import flogging

TILE_SIZE = 512
MAX_SIZE = 512
MAX_SIZE_F = MAX_SIZE*2
PADDING = 2
UPSCALE = 4

def rgb2ycbcr(im):
    xform = np.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
    ycbcr = im.dot(xform.T)
    ycbcr[:,:,[1,2]] += 128
    return ycbcr

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float)
    rgb[:,:,[1,2]] -= 128
    return np.uint8(rgb.dot(xform.T).clip(0.0, 255.0))

def rgb2yuv( rgb ):
     
    m = np.array([[ 0.29900, -0.16874,  0.50000],
                 [0.58700, -0.33126, -0.41869],
                 [ 0.11400, 0.50000, -0.08131]])
     
    yuv = np.dot(rgb,m)
    yuv[:,:,1:]+=128.0
    return yuv

#input is an YUV numpy array with shape (height,width,3) can be uint,int, float or double,  values expected in the range 0..255
#output is a double RGB numpy array with shape (height,width,3), values in the range 0..255
def yuv2rgb( yuv ):
      
    m = np.array([[ 1.0, 1.0, 1.0],
                 [-0.000007154783816076815, -0.3441331386566162, 1.7720025777816772],
                 [ 1.4019975662231445, -0.7141380310058594 , 0.00001542569043522235] ])
    
    rgb = np.dot(yuv,m)
    rgb[:,:,0]-=179.45477266423404
    rgb[:,:,1]+=135.45870971679688
    rgb[:,:,2]-=226.8183044444304
    return np.uint8(rgb.clip(0.0, 255.0))

def getTileImages(image, width=TILE_SIZE, height=TILE_SIZE):
    _nrows, _ncols, depth = image.shape
    _size = image.size

    nrows, _m = divmod(_nrows, height)
    ncols, _n = divmod(_ncols, width)
    if _m != 0 or _n != 0:
        image2 = np.zeros(((nrows+1) * height, (ncols+1) * width, depth), dtype=np.float32)
        image2[:_nrows, :_ncols, :] = image[:, :, :]
        nrows += 1
        ncols += 1
    else:
        image2 = image
    _strides = image2.strides

    return np.lib.stride_tricks.as_strided(
        np.ravel(image2),
        shape=(nrows, ncols, height, width, depth),
        strides=(height * _strides[0], width * _strides[1], *_strides),
        writeable=False
    )

def subdivide(image, max_size=MAX_SIZE, repeat=False):
    height, width, depth = image.shape
    newWidth, newHeight = width, height
    nx, ny = 1, 1

    while newHeight > max_size:
        if newHeight & 1 == 0:
            newHeight >>= 1
        else:
            newHeight >>= 1
            newHeight += 1
        nx <<= 1

    while newWidth > max_size:
        if newWidth & 1 == 0:
            newWidth >>= 1
        else:
            newWidth >>= 1
            newWidth += 1
        ny <<= 1

    paddedImage = np.zeros((nx * newHeight + 2*PADDING, ny * newWidth + 2*PADDING, depth), dtype=np.float32)
    paddedImage[PADDING:(height+PADDING), PADDING:(width+PADDING), :] = image[:, :, :]

    if repeat and height >= PADDING and width >= PADDING:
        ph, pw, pd = paddedImage.shape
        #upper left corner
        paddedImage[0:PADDING, 0:PADDING, :] = image[(height-PADDING):height, (width-PADDING):width, :]
        #lower right corner
        paddedImage[(ph-PADDING):ph, (pw-PADDING):pw, :] = image[0:PADDING, 0:PADDING, :]
        #upper right corner
        paddedImage[0:PADDING, (pw-PADDING):pw, :] = image[(height-PADDING):height, 0:PADDING, :]
        #lower left corner
        paddedImage[(ph-PADDING):ph, 0:PADDING, :] = image[0:PADDING, (width-PADDING):width, :]
        #up side
        paddedImage[0:PADDING, PADDING:(width+PADDING), :] = image[(height-PADDING):height, :, :]
        #down side
        paddedImage[(ph-PADDING):ph, PADDING:(width+PADDING), :] = image[0:PADDING, :, :]
        #left side
        paddedImage[PADDING:(height+PADDING), 0:PADDING, :] = image[:, (width-PADDING):width, :]
        #right side
        paddedImage[PADDING:(height+PADDING), (pw-PADDING):pw, :] = image[:, 0:PADDING, :]

    out = np.zeros((nx, ny, newHeight + 2*PADDING, newWidth + 2*PADDING, depth), dtype=np.float32)

    for i in range(nx):
        for j in range(ny):
            out[i, j, :, :, :] = paddedImage[i*newHeight:((i+1)*newHeight + 2*PADDING), j*newWidth:((j+1)*newWidth + 2*PADDING), :]

    return out

def fillFullImage(subUpscaledImg, image, nx, ny):
    height, width, depth = subUpscaledImg.shape
    uPadding = PADDING * UPSCALE
    tHeight, tWidth = height-2*uPadding, width-2*uPadding

    up = nx*tHeight
    down = min((nx+1)*tHeight, image.shape[0])
    left = ny*tWidth
    right = min((ny+1)*tWidth, image.shape[1])

    image[up:down, left:right, :] = subUpscaledImg[uPadding:uPadding+(down-up), uPadding:uPadding+(right-left), :]

def imageFromTiled(imageShape, imageTile, x, y, out):
    tilesX = imageShape[0]
    tilesY = imageShape[1]
    tilesSizeX = imageShape[2] * 4
    tilesSizeY = imageShape[3] * 4

    out[x*tilesSizeX:(x+1)*tilesSizeX, y*tilesSizeY:(y+1)*tilesSizeY, :] = imageTile[:, :, :]

def m11to255(npData):
    '''Convert -1..1 to 0..255'''
    return ((npData + 1.0) * 127.5).astype(np.uint8)

def m01to255(npData):
    '''Convert 0..1 to 0..255'''
    return (npData * 255.0).astype(np.uint8)

def m11to255Float(npData):
    '''Convert -1..1 to 0..255'''
    return ((npData + 1.0) * 127.5)

def _255tom11(npData):
    '''Convert 0..255 to -1..1'''
    return (npData / 127.5 - 1.0).astype(np.float32)

def _255tom01(npData):
    '''Convert 0..255 to 0..1'''
    return (npData / 255.0).astype(np.float32)

def secToTime(sec):
    return "{:d}:{:02d}:{:02d}".format(int(sec) // 3600, (int(sec) // 60) % 60, int(sec) % 60)

#ltime = time.time()
def convertFrames(reader, writer, args):
    global filter
    try:
        if args['filter'] in filters:
            filter = filters[args['filter']](args)
            if reader.useDifFilter:
                filter = DifFilter(args, filter)
        else:
            args["logger"].error("Filter {} not found".format(args['filter']))
            return
    except FilterArgumentError as e:
        logger = flogging.FilterLogging(args["loglevel"], e.filter)
        logger.error(e.text)
        return
    #ltime = time.time()
    #times = []
    bar = tqdm(total=reader.numFrames*filter.outBatchSize)
    li = 0
    for frame in reader:
        i = reader.readFrames
        #t = time.time() - ltime
        #ltime = time.time()
        #fps = 1.0 / max(t, 0.000001)
        #times.append(t)
        #if len(times) > reader.fps * 10:
        #    times.pop(0)
        #mean_t = sum(times)/len(times)
        #print("Upscaling frame {}, {}, {:.3f} FPS, {} seconds remainings".format(i, secToTime(i / reader.fps), fps, secToTime(mean_t * (reader.numFrames - i - 1))))
        bar.update(i-li)
        li = i

        out = filter.process(frame, i)
        if not out is None:
            writer.write(out)

    logger = flogging.FilterLogging(args["loglevel"], "converter")
    logger.print("Converted!")
