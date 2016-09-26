import os
import errno

def create_dir(dir_path):
    if not os.path.exists(os.path.dirname(dir_path)):
        try:
            os.makedirs(os.path.dirname(dir_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise