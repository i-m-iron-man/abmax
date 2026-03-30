import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'


from abmax.functions import *
import jax.numpy as jnp
import jax.random as random
import jax

from sim_params import *
from agent import *
from neuron_policy import *
from ray_sensing import *
from utils import *
from train import *


def test_one_param(ES_Params, boid_test_world):

    boid_test_world = jit_set_CMAES_params(ES_Params, boid_test_world)# ES_Params size should be (num_boids, num_params)
    
    boid_world, total_age_series, num_active_agents_series, render_data = jit_run_episode(boid_test_world) #already has reset_world
    
    total_age_series_sum = jnp.sum(total_age_series) # sum total age across scenarios for each world, shape: (NUM_WORLDS, NUM_SCENARIOS)
    age_fitness = jnp.mean(total_age_series_sum/(EP_LEN*NUM_AGENTS*NUM_AGENTS))
    
    total_grazing_energy_sum = jnp.sum(boid_test_world.boid_set.agents.state.content['grazing_energy_sum'].reshape(-1)) # shape: (1)
    total_metabolic_cost_sum = jnp.sum(boid_test_world.boid_set.agents.state.content['metabolic_cost_sum'].reshape(-1)) # shape: (1)
    energy_fitness = (total_grazing_energy_sum - total_metabolic_cost_sum)/EP_LEN # compute fitness for each world and scenario, shape: (NUM_WORLDS, NUM_SCENARIOS, 1)
    
    fitness = age_fitness + 0.5*energy_fitness
    final_js = boid_world.boid_set.agents.policy.params.content['J']
    
    return render_data, num_active_agents_series, final_js, fitness
jit_test_one_param = jax.jit(test_one_param)

def test_all_params(ES_Params, boid_test_world, param_type_str):
    
    #num_test_worlds = ES_Params.shape[0]
    #sim_params = Params(content=SIM_PARAMS_CONTENT)

    #key, *subkeys = jax.random.split(key, num_test_worlds+1)
    #subkeys = jnp.array(subkeys)
    
    #boid_worlds = jax.vmap(Boid_world.create_world, in_axes=(None, 0))(sim_params, subkeys) # each world for each fitness level

    render_data, num_active_agents_series, final_js, fitness = jax.vmap(jit_test_one_param, in_axes=(0, None))(ES_Params, boid_test_world)

    boid_xs = render_data.content['xs']
    boid_ys = render_data.content['ys']
    boid_angs = render_data.content['angs']
    boid_energies = render_data.content['energies']
    boid_grazing_energies = render_data.content['grazing_energies']
    boid_exchange_energies = render_data.content['exchange_energies']
    boid_metabolic_costs = render_data.content['metabolic_costs']
    boid_roles = render_data.content['roles']
    boid_num_active_agents_series = num_active_agents_series.reshape(-1)
    boid_final_js = final_js
    fitness = jnp.array(fitness)

    path = TRAJ_PATH + f"{param_type_str}/"
    
    jnp.save(path + "rendering_boids_xs.npy", boid_xs)
    jnp.save(path + "rendering_boids_ys.npy", boid_ys)
    jnp.save(path + "rendering_boids_angs.npy", boid_angs)
    jnp.save(path + "rendering_boids_energies.npy", boid_energies)
    jnp.save(path + "rendering_boids_grazing_energies.npy", boid_grazing_energies)
    jnp.save(path + "rendering_boids_exchange_energies.npy", boid_exchange_energies)
    jnp.save(path + "rendering_boid_metabolic_costs.npy", boid_metabolic_costs)
    jnp.save(path + "rendering_boid_roles.npy", boid_roles)
    jnp.save(path + "rendering_boid_num_active_agents_series.npy", boid_num_active_agents_series)
    jnp.save(path + "test_fitness.npy", fitness)
    jnp.save(path + "rendering_boid_final_js.npy", boid_final_js)

if __name__ == "__main__":
    ES_Params_mean = jnp.load(PARAM_PATH+'params_list_mean.npy') # mean params from fitness 19.5 to 1.5
    #ES_Params_best = jnp.load(PARAM_PATH+'params_list_best.npy') # best params from fitness 19.5 to 1.5

    key = jnp.load(PARAM_PATH+'test_key.npy')
    key, test_key = jax.random.split(key)

    sim_params = Params(content=SIM_PARAMS_CONTENT)
    boid_test_world = Boid_world.create_world(sim_params, test_key)
    
    #key, swarm_key = jax.random.split(key)
    test_all_params(ES_Params_mean, boid_test_world, 'mean')
    #test_all_params(ES_Params_best, boid_test_world, 'best')