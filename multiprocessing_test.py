import numpy as np
import multiprocessing as mp
import sys

def f(name):
    print(name)

if __name__ == '__main__':
    mp.freeze_support()
    p = mp.Process(target=f, args=('bob',))
    p.start()
    p.join()
    # make sure all output has been processed before we exit
    sys.stdout.flush()


"""
def diag(a):
    print(a)

mp.set_start_method('fork')
processes = []
for i in np.arange(10):
    p = Process(target=diag, args=(i,))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
"""
