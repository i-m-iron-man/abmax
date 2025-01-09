import timeit
import gc

setup = """
gc.enable()
import os, sys
import jax
import jax.numpy as jnp

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

key, *ecosystem_keys = jax.random.split(key, 11)
ecosystem_keys = jnp.array(ecosystem_keys)
    
ecosystems = jax.vmap(Ecosystem.create_ecossystem, 
                      in_axes=(None, None, None, None, None, None, None, None, None, 0))(grass_regrowth_time, wolf_reproduction_probab, wolf_energy, init_wolves, sheep_reproduction_probab, sheep_energy, init_sheeps, space_size, sim_steps, ecosystem_keys)
ts = jnp.arange(sim_steps)
    
def run():
    new_ecosystems, num_agents = jax.vmap(Ecosystem.run_loop, in_axes=(0, None))(ecosystems, ts)

"""

n_run = 10
tt = timeit.Timer("run()", setup=setup.format())

a = tt.repeat(n_run, 1)
median_time = sorted(a)[n_run // 2 + n_run % 2]
print("abmax WolfSheep-small (ms):", median_time*1e3)