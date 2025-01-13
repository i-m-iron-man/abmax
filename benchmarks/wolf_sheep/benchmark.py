import timeit
import gc

setup_small = """
gc.enable()
import os, sys
import jax

from sim import Ecosystem, ecosystem_small

def run_small():
    new_ecosystem, num_agents = Ecosystem.run(ecosystem_small)
"""

n_run = 10
tt_small = timeit.Timer("run_small()", setup=setup_small.format())

a_small = tt_small.repeat(n_run, 1)
median_time = sorted(a_small)[n_run // 2 + n_run % 2]
print("abmax WolfSheep-small (ms):", median_time*1e3)

setup_large = """
gc.enable()
import os, sys
import jax

from sim import Ecosystem, ecosystem_large

def run_large():
    new_ecosystem, num_agents = Ecosystem.run(ecosystem_large)
"""

n_run = 10
tt_large = timeit.Timer("run_large()", setup=setup_large.format())

a_large = tt_large.repeat(n_run, 1)
median_time = sorted(a_large)[n_run // 2 + n_run % 2]
print("abmax WolfSheep-large (ms):", median_time*1e3)