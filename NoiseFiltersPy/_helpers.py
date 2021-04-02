import time

def timeit(function, *args, **params):
    tstart = time.time()
    result = function(*args, **params)
    tend = time.time()

    return result, tend - tstart

