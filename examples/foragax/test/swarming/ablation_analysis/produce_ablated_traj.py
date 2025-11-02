
import jax.numpy as jnp
import jax.random as random
from abmax.structs import *
from abmax.functions import *

from test_env import *

seeds = [11,7,5]
i=1 # i = 0,1,2 for different seeds
seed = seeds[i]

TEST_RENDER_PATH = "./trajectories_s_" + str(seed) + "/ablated/"

TEST_MAX_SPAWN_X = 100.0
TEST_MAX_SPAWN_Y = 100.0
TEST_NUM_FORAGERS = 10

NUM_TEST_WORLDS = 5


TEST_FORAGING_WORLD_PARAMS = Params(content = {
"forager_params":{
    "x_max": TEST_MAX_SPAWN_X,
    "y_max": TEST_MAX_SPAWN_Y,
    "energy_begin_max": FORAGER_ENERGY_BEGIN_MAX,
    "radius": FORAGER_RADIUS,
    "agent_type": FORAGER_AGENT_TYPE,
    "num_foragers": TEST_NUM_FORAGERS
},
"policy_params":{
    "num_neurons": NUM_NEURONS,
    "num_obs": NUM_OBS,
    "num_actions": NUM_ACTIONS,
    "init_param_span": INIT_PARAM_SPAN
}})


def get_ablated_sensor_data(foragers):
    agent_xs = foragers.state.content['x'].reshape(-1)
    agent_ys = foragers.state.content['y'].reshape(-1)
    agent_rads = foragers.params.content['radius'].reshape(-1)
    agent_energies = foragers.state.content['energy'].reshape(-1)
    agent_types = foragers.agent_type.reshape(-1)  # 1 for forager, 2 for patch

    points = jax.vmap(Point)(agent_xs, agent_ys)
    circles = jax.vmap(Circle)(points, agent_rads)

    type_sensor_values = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # empty, forager, patch, used with lax.select

    def for_each_forager(forager):
        forager_pos = (forager.state.content['x'][0], forager.state.content['y'][0], forager.state.content['ang'][0])
        rays = generate_rays(forager_pos, RAY_SPAN, RAY_MAX_LENGTH)

        def for_each_ray(ray):
            intercepts =  jax.vmap(jit_get_ray_circle_collision, in_axes=(None, 0))(ray, circles)
            min_dist_indx = jnp.argmin(intercepts)
            
            min_dist = ray.length#intercepts[min_dist_indx]
            '''
            sensed_energy, sensed_type = jax.lax.cond(min_dist<ray.length, 
                                                      lambda _: (agent_energies[min_dist_indx], agent_types[min_dist_indx]),
                                                      lambda _: (0.0, 0),  # 0 for empty, 1 for forager, 2 for patch
                                                      None)
            
            '''
            sensed_energy = 0.0
            sensed_type = 0
            sensed_type_value = type_sensor_values[sensed_type]  # get the type sensor value based on the sensed type

            sensed_energy_channels = sensed_energy*sensed_type_value  # energy is only sensed if the type is forager or patch

            return jnp.concatenate((jnp.array([min_dist]), sensed_energy_channels))
        
        return jax.vmap(for_each_ray)(rays).reshape(-1)
    return jax.vmap(for_each_forager)(foragers)
jit_get_ablated_sensor_data = jax.jit(get_ablated_sensor_data)


def agent_ablated_interactions(foragers:Forager):
    
    def forager_forager_interaction(forager, foragers):
        xs_foragers = foragers.state.content['x']  # an array of shape (num_foragers,)
        ys_foragers = foragers.state.content['y']
        ids_foragers = foragers.id  # an array of shape (num_foragers,)

        x_forager = forager.state.content['x']  # a 1x1 array
        y_forager = forager.state.content['y']
        id_forager = forager.id  # a 1x1 array
        
        dist = jnp.linalg.norm(jnp.stack((xs_foragers - x_forager, ys_foragers - y_forager), axis=1), axis=1).reshape(-1)
        cond = jnp.logical_and(dist < forager.params.content['radius'], ids_foragers != id_forager)  # True if the forager is near another forager and not itself
        is_near = jnp.where(cond, 1.0, 0.0)  # 1 if the forager is near another forager, 0 otherwise
        is_in_forager = 0#jnp.sum(is_near)  # how many other foragers the forager is in

        dist = jnp.where(ids_foragers != id_forager, dist, MAX_SPAWN_X)
        nn_dist = jnp.min(dist)

        return is_in_forager, nn_dist

    is_in, nn_dists = jax.vmap(forager_forager_interaction, in_axes=(0, None))(foragers, foragers)

    is_in = is_in.reshape(-1)
    return is_in, nn_dists

jit_agent_ablated_interactions = jax.jit(agent_ablated_interactions)




@struct.dataclass
class Foraging_world():
    forager_set: Set
    @staticmethod
    def create_foraging_world(params, key):
        forager_params = params.content['forager_params']
        policy_params = params.content['policy_params']

        num_foragers = forager_params['num_foragers']

        key, *policy_keys = random.split(key, num_foragers + 1)
        policy_keys = jnp.array(policy_keys)

        policy_create_params = Params(content={ 'num_neurons': policy_params['num_neurons'],
                                                'num_obs': policy_params['num_obs'],
                                                'num_actions': policy_params['num_actions'],
                                                'init_param_span': policy_params['init_param_span']})
        policies = jax.vmap(CTRNN.create_policy, in_axes=(None, 0))(policy_create_params, policy_keys)

        key, forager_key = random.split(key, 2)
        
        x_max_array = jnp.tile(jnp.array([forager_params['x_max']]), (num_foragers,))
        y_max_array = jnp.tile(jnp.array([forager_params['y_max']]), (num_foragers,))
        energy_begin_max_array = jnp.tile(jnp.array([forager_params['energy_begin_max']]), (num_foragers,))
        radius_array = jnp.tile(jnp.array([forager_params['radius']]), (num_foragers,))

        forager_create_params = Params(content={ 'x_max': x_max_array, 
                                                 'y_max': y_max_array, 
                                                 'energy_begin_max': energy_begin_max_array,
                                                 'radius': radius_array,
                                                 'policy': policies})
        
        foragers = create_agents(agent=Forager, params=forager_create_params, num_agents=num_foragers, num_active_agents= num_foragers, 
                                 agent_type = forager_params['agent_type'], key=forager_key)
        forager_set =  Set(num_agents=num_foragers, num_active_agents=num_foragers, agents=foragers, id=0, set_type=forager_params['agent_type'], 
                           params=None, state=None, policy=None, key=None)

        return Foraging_world(forager_set=forager_set)
    
def step_world(foraging_world, _t):
    forager_set = foraging_world.forager_set    
    is_in_sum, NN_dist = jit_agent_ablated_interactions(forager_set.agents)
    avg_NN_dist = jnp.mean(NN_dist)
    
    sensor_data = jit_get_ablated_sensor_data(forager_set.agents)
    

    forager_step_input = Signal(content={'obs': sensor_data, 'energy_in': jnp.zeros(TEST_NUM_FORAGERS), 
                                         'is_in_sum': is_in_sum})

    forager_step_params = Params(content={'dt': Dt, 
                                            'damping': DAMPING, 
                                            'metabolic_cost_speed': 0.0,#METABOLIC_COST_SPEED,
                                            'metabolic_cost_angular': 0.0,#METABOLIC_COST_ANGULAR,
                                            'x_max_arena': MAX_WORLD_X, 
                                            'y_max_arena': MAX_WORLD_Y,
                                            'edge_penalty': EDGE_PENALTY,
                                            'action_scale': ACTION_SCALE,
                                            'time_constant_scale': TIME_CONSTANT_SCALE
                                            })
        
    forager_set = jit_step_agents(Forager.step_agent, forager_step_params, forager_step_input, forager_set)

    #also collect the neuron activation for 1st forager for visualization

    render_data = Signal(content={
        'forager_xs': forager_set.agents.state.content['x'].reshape(-1, 1),
        'forager_ys': forager_set.agents.state.content['y'].reshape(-1, 1),
        'forager_angs': forager_set.agents.state.content['ang'].reshape(-1, 1),
        'forager_energies': forager_set.agents.state.content['energy'].reshape(-1, 1),
        'avg_NN_dist': avg_NN_dist,
        'forager_zs': forager_set.agents.policy.state.content['Z'][0] # just the first forager
    }) # it was not exceeding GPU memory without render_data, but it may now!


    return foraging_world.replace(forager_set=forager_set), render_data

jit_step = jax.jit(step_world)

def reset_world(foraging_world):
    foraging_set_agents = foraging_world.forager_set.agents

    foraging_set_agents = jax.vmap(Forager.reset_agent)(foraging_set_agents, None)

    forager_set = foraging_world.forager_set.replace(agents=foraging_set_agents)

    return foraging_world.replace(forager_set=forager_set)

jit_reset_world = jax.jit(reset_world)

def scan_episode(foraging_world:Foraging_world, ts):
    """
    Scan the foraging world for a given number of time steps.
    Args:
        - foraging_world: The foraging world to scan. this is the carry value
        - ts: The time steps to scan
    Returns:
        - The updated foraging world after the time steps
        - the render_data for the foragers and the patches: (forsger_xs: (num_foragers, ts), forager_ys: (num_foragers, ts), forager_angs: (num_foragers, ts), patch_energies: (num_patches, ts))
    """
    return jax.lax.scan(jit_step, foraging_world, ts) # scan(Scanning_function, carry, xs) where xs is the time steps to scan
jit_scan_episode = jax.jit(scan_episode)

def run_episode(foraging_world:Foraging_world):
    """
    run the episode. This function is used as a wrapper for the scan_episode function so that a jitted version can be created
    Args:
        - foraging_world: The foraging world to run the episode
    Returns:
        - The updated foraging world after the episode
        - the render_data for the foragers and the patches: (forsger_xs: (num_foragers, ts), forager_ys: (num_foragers, ts), forager_angs: (num_foragers, ts), patch_energies: (num_patches, ts))
    """
    ts = jnp.arange(EP_LEN)
    foraging_world = jit_reset_world(foraging_world)  # reset the world before running the episode
    foraging_world, render_data = jit_scan_episode(foraging_world, ts)
    return foraging_world, render_data

jit_run_episode = jax.jit(run_episode)

def test_one_param(ES_Params, foraging_world):

    ES_Params = jnp.tile(ES_Params, (NUM_FORAGERS, 1)) # ES_Params shape (num_foragers, num_params)
    es_foragers = jit_set_CMAES_params(ES_Params, foraging_world.forager_set.agents)# ES_Params size should be (num_foragers, num_params)
    es_forager_set = foraging_world.forager_set.replace(agents=es_foragers)
    foraging_world = foraging_world.replace(forager_set=es_forager_set)

    foraging_world, rendering_data = jit_run_episode(foraging_world) #already has reset_world

    fitness = foraging_world.forager_set.agents.state.content['fitness'].reshape(-1)
    
    mean_fitness = jnp.mean(fitness)
    max_fitness = jnp.max(fitness)
    min_fitness = jnp.min(fitness)
    fitness = jnp.array([mean_fitness, max_fitness, min_fitness])

    return rendering_data, fitness
jit_test_one_param = jax.jit(test_one_param)

def test_param(ES_Params, key):
    
    num_test_worlds = NUM_TEST_WORLDS #ES_Params.shape[0]
    key, *subkeys = jax.random.split(key, num_test_worlds+1)
    subkeys = jnp.array(subkeys)
    foraging_worlds = jax.vmap(Foraging_world.create_foraging_world, in_axes=(None, 0))(TEST_FORAGING_WORLD_PARAMS, subkeys)

    render_data, fitness = jax.vmap(jit_test_one_param, in_axes=(None, 0))(ES_Params, foraging_worlds)

    foragers_xs = render_data.content['forager_xs']
    foragers_ys = render_data.content['forager_ys']
    foragers_angs = render_data.content['forager_angs']
    forager_energies = render_data.content['forager_energies']
    forager_avg_NN_dists = render_data.content['avg_NN_dist']

    jnp.save(TEST_RENDER_PATH + "rendering_foragers_xs.npy", foragers_xs)
    jnp.save(TEST_RENDER_PATH + "rendering_foragers_ys.npy", foragers_ys)
    jnp.save(TEST_RENDER_PATH + "rendering_foragers_angs.npy", foragers_angs)
    jnp.save(TEST_RENDER_PATH + "rendering_forager_energies.npy", forager_energies)
    jnp.save(TEST_RENDER_PATH + "rendering_forager_avg_NN_dists.npy", forager_avg_NN_dists)

if __name__ == "__main__":
    ES_Params = jnp.load("../../params/seed_" + str(seed) + "/params_list.npy")
    #remove the last  params set as it is ineffective saved because the in the last generation
    ES_Params = ES_Params[:-1,:]
    ES_Params = ES_Params[-1:,:] # just the last set of params
    key = jnp.load("../../params/seed_" + str(seed) + "/test_key.npy")
    key, forage_key = jax.random.split(key)
    key, swarm_key = jax.random.split(key)
    test_param(ES_Params, swarm_key)