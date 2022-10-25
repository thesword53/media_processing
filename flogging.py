LOGLEVEL_DEBUG = 0
LOGLEVEL_VERBOSE = 1
LOGLEVEL_WARNING = 2
LOGLEVEL_ERROR = 3
LOGLEVEL_FATAL = 4

class FilterLogging:
    def __init__(self, currentLevel=LOGLEVEL_VERBOSE, name=None):
        self.currentLevel = currentLevel
        self.name = name

    def log(self, level, text, *fmt):
        textFmt = text.format(*fmt) 

        textName = ""
        if self.name:
            textName = self.name + ":"

        if level >= self.currentLevel:
            if level == LOGLEVEL_DEBUG:
                print("\033[92m[{:s}DEBUG] {:s}\033[0m".format(textName, textFmt))
            elif level == LOGLEVEL_WARNING:
                print("\033[93m[{:s}WARNING] {:s}\033[0m".format(textName, textFmt))
            elif level == LOGLEVEL_ERROR:
                print("\033[91m[{:s}ERROR] {:s}\033[0m".format(textName, textFmt))
            elif level == LOGLEVEL_FATAL:
                print("\033[31m[{:s}FATAL] {:s}\033[0m".format(textName, textFmt))
            else:
                if self.name:
                    print("\033[97m[{:s}]\033[0m {:s}".format(self.name, textFmt))
                else:
                    print(textFmt)

    def debug(self, text, *fmt):
        self.log(LOGLEVEL_DEBUG, text, *fmt)

    def print(self, text, *fmt):
        self.log(LOGLEVEL_VERBOSE, text, *fmt)

    def warn(self, text, *fmt):
        self.log(LOGLEVEL_WARNING, text, *fmt)

    def error(self, text, *fmt):
        self.log(LOGLEVEL_ERROR, text, *fmt)

    def fatal(self, text, *fmt):
        self.log(LOGLEVEL_FATAL, text, *fmt)