import timeit
import gc

setup = """
gc.enable()
import os, sys
from sim import Bird, main, bird_set, num_steps


def run():
    xs,ys = main(bird_set, num_steps)

"""
n_run = 10
tt = timeit.Timer("run()", setup=setup.format())

a = tt.repeat(n_run, 1)
median_time = sorted(a)[n_run // 2 + n_run % 2]
print("abmax Flocking-small (ms):", median_time*1e3)

"""
params_content = {'X_pos_max': jnp.tile(X_pos_max, num_agents),
                  "Y_pos_max": jnp.tile(Y_pos_max, num_agents),
                  "speed": jnp.tile(speed, num_agents),
                  "cohere_factor": jnp.tile(cohere_factor, num_agents),
                  "seperate_factor": jnp.tile(separate_factor, num_agents),
                  "match_factor": jnp.tile(match_factor, num_agents),
                  "visual_distance": jnp.tile(visual_distance, num_agents),
                  "seperation": jnp.tile(seperation, num_agents)}
params = Params(content=params_content)

bird_agents = create_agents(Bird, params, num_agents, num_active_agents, agent_type, key)
bird_set = Set(agents=bird_agents, num_agents=num_agents, num_active_agents=num_active_agents,
               id=0, set_type=0, params=None, state=None, policy=None, key=None)
def run():    
    xs,ys = run_loop(bird_set, num_steps)
"""