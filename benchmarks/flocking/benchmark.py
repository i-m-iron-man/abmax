import timeit
import gc

setup_small = """
gc.enable()
import os, sys
from sim import Bird, main, bird_set_small
num_steps = 100

def run_small():
    xs,ys = main(bird_set_small, num_steps)

"""
n_run = 10
tt = timeit.Timer("run_small()", setup=setup_small.format())

a = tt.repeat(n_run, 1)
median_time = sorted(a)[n_run // 2 + n_run % 2]
print("abmax Flocking-small (ms):", median_time*1e3)



setup_large = """
gc.enable()
import os, sys
from sim import Bird, main, bird_set_large
num_steps = 100

def run_large():
    xs,ys = main(bird_set_large, num_steps)

"""
n_run = 10
tt = timeit.Timer("run_large()", setup=setup_large.format())

a = tt.repeat(n_run, 1)
median_time = sorted(a)[n_run // 2 + n_run % 2]
print("abmax Flocking-large (ms):", median_time*1e3)