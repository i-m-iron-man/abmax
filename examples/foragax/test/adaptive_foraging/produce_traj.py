from abmax.structs import *
from abmax.functions import *


from test_env import *
seeds = [11,7,5]
i=0 # i = 0,1,2 for different seeds
seed = seeds[i]

TEST_RENDER_PATH = "./trajectories_s_" + str(seed) + "/"

def test_one_param(ES_Params, foraging_world):

    ES_Params = jnp.tile(ES_Params, (NUM_FORAGERS, 1)) # ES_Params shape (num_foragers, num_params)
    es_foragers = jit_set_CMAES_params(ES_Params, foraging_world.forager_set.agents)# ES_Params size should be (num_foragers, num_params)
    es_forager_set = foraging_world.forager_set.replace(agents=es_foragers)
    foraging_world = foraging_world.replace(forager_set=es_forager_set)

    foraging_world, rendering_data = jit_run_episode(foraging_world) #already has reset_world

    fitness = foraging_world.forager_set.agents.state.content['energy'].reshape(-1)
    
    mean_fitness = jnp.mean(fitness)
    max_fitness = jnp.max(fitness)
    min_fitness = jnp.min(fitness)
    fitness = jnp.array([mean_fitness, max_fitness, min_fitness])

    return rendering_data, fitness

jit_test_one_param = jax.jit(test_one_param)

def test_all_params(ES_Params, key):
    
    #num_test_worlds = ES_Params.shape[0]
    key, subkey = jax.random.split(key, 2)

    #foraging_worlds = jax.vmap(Foraging_world.create_foraging_world, in_axes=(None, 0))(FORAGING_WORLD_PARAMS, subkeys)
    foraging_world = Foraging_world.create_foraging_world(FORAGING_WORLD_PARAMS, subkey)

    render_data, fitness = jax.vmap(jit_test_one_param, in_axes=(0, None))(ES_Params, foraging_world)

    foragers_xs = render_data.content['forager_xs']
    foragers_ys = render_data.content['forager_ys']
    foragers_angs = render_data.content['forager_angs']
    foragers_energies = render_data.content['forager_energies']
    patch_energies = render_data.content['patch_energies']
    patch_xs = render_data.content['patch_xs']
    patch_ys = render_data.content['patch_ys']

    jnp.save(TEST_RENDER_PATH + "rendering_foragers_xs.npy", foragers_xs)
    jnp.save(TEST_RENDER_PATH + "rendering_foragers_ys.npy", foragers_ys)
    jnp.save(TEST_RENDER_PATH + "rendering_foragers_angs.npy", foragers_angs)
    jnp.save(TEST_RENDER_PATH + "rendering_foragers_energies.npy", foragers_energies)
    jnp.save(TEST_RENDER_PATH + "rendering_patch_energies.npy", patch_energies)
    jnp.save(TEST_RENDER_PATH + "rendering_patch_xs.npy", patch_xs)
    jnp.save(TEST_RENDER_PATH + "rendering_patch_ys.npy", patch_ys)
    jnp.save(TEST_RENDER_PATH + "fitness.npy", fitness)


if __name__ == "__main__":
    # change the seed here to get different random params

    ES_Params = jnp.load("../../params/seed_" + str(seed) + "/params_list.npy")
    ES_Params = ES_Params[:-1,:] # remove last param set which may be suboptimal

    key = jnp.load("../../params/seed_" + str(seed) + "/test_key.npy")
    key, forage_key = jax.random.split(key)
    test_all_params(ES_Params, forage_key)
