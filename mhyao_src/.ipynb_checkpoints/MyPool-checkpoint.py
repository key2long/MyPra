import multiprocessing
import multiprocessing.pool
import pdb
import traceback
import sys
import time
import random

class NoDaemonProcess(multiprocessing.Process):
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess


def print_process_id(parent_id: int,
                     child_id: int):
    print(f"parent_id:{parent_id};"
          f"child_id:{child_id}.")


def open_process(parent_id: int):
    p_num = 3
    try:
        child_pool = multiprocessing.Pool(processes=p_num)
        for j in range(p_num):
            child_pool.apply_async(func=print_process_id,
                                   args=(parent_id, j))
        child_pool.close()
        child_pool.join()
    except:
        error = traceback.format_exc()
        raise Exception(error)
    start = time.time()
    time.sleep(random.random() * 3)
    end = time.time()
    print(f'parent id:{parent_id} takes:{end-start}')
    


def handle_error(error: str):
    print(error)
    sys.stdout.flush()