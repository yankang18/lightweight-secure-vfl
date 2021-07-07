import os
import shutil

def get_logger():
    return SimpleLogger()


class SimpleLogger(object):

    def trace(self, msg):
        print("===> [TRACE] " + msg)

    def debug(self, msg):
        print("[DEBUG] " + msg)

    def info(self, msg):
        print("[INFO] " + msg)

def create_exp_dir(path, scripts_to_save=None):
    os.makedirs(path, exist_ok=True)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


