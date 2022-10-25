from collections import OrderedDict

def argsToDict(argsStr):
    argDict = OrderedDict()
    if len(argsStr) > 0:
        argsStrList = argsStr.split(';')
        for arg in argsStrList:
            elts = arg.split('=', 1)
            if len(elts) >= 2:
                argDict[elts[0]] = elts[1]
            else:
                argDict[elts[0]] = None
    return argDict

color_profiles = {
    'bt709': {
        'out_color_matrix': 'bt709',
        'color_primaries': 'bt709',
        'color_trc': 'bt709',
        'colorspace': 'bt709'
    }
}

encoders_profiles = {
    'h264_nvenc': {
        'cq': '30',
        'high': {
            'profile': 'high',
            'pix_fmt': ('yuv420p', 'nv12', 'p010le', 'p016le', 'bgr0', 'rgb0'),
            'cq': '30'
        }
    }
}

def getArgumentsProfile(profile, argsStr):
    args = argsToDict(argsStr)