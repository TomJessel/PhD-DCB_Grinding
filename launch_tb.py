#!/usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorboard import program
from pathlib import PurePosixPath as Path
import sys


PLATFORM = os.name
if PLATFORM == 'nt':
    onedrive = Path(
        r'C:\Users\tomje\OneDrive - Cardiff University\Documents\PHD\AE'
    )
    TB_DIR = onedrive.joinpath('Tensorboard')
elif PLATFORM == 'posix':
    onedrive = Path(
        r'/mnt/c/Users/tomje/OneDrive - Cardiff University/Documents/PHD/AE'
    )
    TB_DIR = onedrive.joinpath('Tensorboard')


def launch_tb(log_dir):
    """
    Launches TensorBoard in a new process, at relative path to TB_DIR.

    If run in jupyter or ipython, will not exit restart kernel.

    Args:
        log_dir (str): Relative path to TB_DIR.
    """

    log_dir = TB_DIR.joinpath(log_dir)
    log_dir = str(log_dir)
    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir, '--load_fast', 'false'])
    url = tb.launch()
    print(f'Logdir: {log_dir}')
    print(f"TensorBoard started at:\n  {url}")
    try:
        get_ipython().__class__.__name__  # type: ignore
        pass
    except NameError:
        try:
            input('Press CRTL+C to exit\n')
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    if len(sys.argv) > 1:
        launch_tb(sys.argv[1])
    else:
        launch_tb(input('Logdir: '))
