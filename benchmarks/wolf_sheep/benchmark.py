import timeit
import gc

setup = """
gc.enable()
import os, sys
import jax

from sim import Ecosystem, run_loop

grass_regrowth_time = 30
space_size = 100
wolf_reproduction_probab = 0.05
wolf_energy = 20
init_wolves = 20
sheep_reproduction_probab = 0.04
sheep_energy = 10
init_sheeps = 100
sim_steps = 100
key = jax.random.PRNGKey(0)

ecosystem = Ecosystem.create_ecossystem(grass_regrowth_time, wolf_reproduction_probab, wolf_energy, init_wolves, 
                                            sheep_reproduction_probab, sheep_energy, init_sheeps, space_size, sim_steps, key)

def run():
    num_agents = run_loop(ecosystem)
"""

n_run = 10
tt = timeit.Timer("run()", setup=setup.format())

a = tt.repeat(n_run, 1)
median_time = sorted(a)[n_run // 2 + n_run % 2]
print("abmax WolfSheep-small (ms):", median_time*1e3)