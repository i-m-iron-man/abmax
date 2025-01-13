import timeit
import gc



setup = """
gc.enable()
import os, sys

import jax
import jax.numpy as jnp

from sim import Ecosystem, ecosystem_vmap

print("starting sim")    
def run():
    new_ecosystems, num_agents = Ecosystem.run_vmap(ecosystem_vmap)


"""

n_run = 10
tt = timeit.Timer("run()", setup=setup.format())

a = tt.repeat(n_run, 1)
median_time = sorted(a)[n_run // 2 + n_run % 2]
print("abmax WolfSheep-vmap (ms):", median_time*1e3)