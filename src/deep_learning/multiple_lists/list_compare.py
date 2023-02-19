import os
from timeit import timeit

import numpy as np
import psutil
import torch

N_ELEMENTS = 100000000


def mutliple_for_loops():

    tensor_buffer = torch.zeros((3, N_ELEMENTS), device=torch.device("cuda"))
    a = []
    b = []
    c = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for i in range(N_ELEMENTS):
        a.append(i)
    tensor_buffer[0] = torch.tensor(a)
    del a
    for i in range(N_ELEMENTS):
        b.append(i * i / (N_ELEMENTS // 2))
    tensor_buffer[1] = torch.tensor(b)
    del b
    for i in range(N_ELEMENTS):
        c.append(-i)
    tensor_buffer[2] = torch.tensor(c)
    del c
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(start.elapsed_time(end))  # milliseconds
    print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)


def single_for_loop():

    tensor_buffer = torch.zeros((3, N_ELEMENTS), device=torch.device("cuda"))
    a = []
    b = []
    c = []

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    for i in range(N_ELEMENTS):
        a.append(i)
        b.append(i * i / (N_ELEMENTS // 2))
        c.append(-i)
    tensor_buffer[0] = torch.tensor(a)
    tensor_buffer[1] = torch.tensor(b)
    tensor_buffer[2] = torch.tensor(c)
    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(start.elapsed_time(end))  # milliseconds
    print(psutil.Process(os.getpid()).memory_info().rss / 1024**2)


def main():
    # print("adsfsdf", timeit(mutliple_for_loops, number=1))
    # print("a", timeit(mutliple_for_loops, number=1))
    print("a", timeit(single_for_loop, number=1))


if __name__ == "__main__":
    main()
