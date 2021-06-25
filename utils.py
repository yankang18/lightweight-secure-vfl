def get_logger():
    return SimpleLogger()


class SimpleLogger(object):

    def trace(self, msg):
        print("===> [TRACE] " + msg)

    def debug(self, msg):
        print("[DEBUG] " + msg)

    def info(self, msg):
        print("[INFO] " + msg)



