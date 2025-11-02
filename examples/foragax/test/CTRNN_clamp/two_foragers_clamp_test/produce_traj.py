from abmax.structs import *
from abmax.functions import *

import jax
import jax.numpy as jnp
import jax.random as random


from test_env import *

import jax.numpy as jnp
import jax.random as random

from test_env import *

TEST_MAX_SPAWN_X = 100.0
TEST_MAX_SPAWN_Y = 100.0

TEST_NUM_FORAGERS = 2

NUM_TEST_WORLDS = 5 # there is still variance in position of foragers due to random key

seeds = [11,7,5]
i=2 # i = 0,1,2 for different seeds

seed = seeds[i]

forager_begin_energy_choice = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
j = 0 # change this index to change the initial energy of foragers

clamped_state = [True, False] # first forager clamped, second not clamped
clamped_state_str = ["clamped", "unclamped"]
k = 0 # change this index to change which forager is clamped

TEST_FORAGER_ENERGY_BEGIN_MAX = forager_begin_energy_choice[j] # max energy at the beginning

TEST_RENDER_PATH = "./trajectories_s_" + str(seed) + "/e_"+str(int(TEST_FORAGER_ENERGY_BEGIN_MAX))+"/" + clamped_state_str[k] + "/"

CLAMPED_FORAGER_ENERGY_BEGIN_MAX = 15.0

special_neuron_seed_7 = [28] # ablated-> 1.0 # needs to be changed by commenting uncommenting in step_policy function line 117
special_neuron_seed_11 = [30, 34] # ablated 30->1.0, 34-> 1.0 # needs to be changed by commenting uncommenting in step_policy function line 118, 119
special_neuron_seed_5 = [17] # ablated 17 -> 1.0 # needs to be changed by commenting uncommenting in step_policy function line 120

TEST_FORAGING_WORLD_PARAMS = Params(content = {
"forager_params":{
    "x_max": TEST_MAX_SPAWN_X,
    "y_max": TEST_MAX_SPAWN_Y,
    "energy_begin_max": TEST_FORAGER_ENERGY_BEGIN_MAX,
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


@struct.dataclass
class CTRNN(Policy):
    @staticmethod
    def create_policy(params:Params, key:jax.random.PRNGKey):
        num_neurons = params.content["num_neurons"]
        num_obs = params.content["num_obs"]
        num_actions = params.content["num_actions"]
        init_param_span = params.content["init_param_span"]

        Z =  jnp.zeros((num_neurons,),dtype=jnp.float32)
        action = jnp.zeros((num_actions,),dtype=jnp.float32)
        
        state = State(content={'Z':Z, 'action':action})
       
        key, *init_keys = jax.random.split(key, 6)

        J = jax.random.uniform(init_keys[0], (num_neurons,num_neurons), minval=-init_param_span, maxval=init_param_span, dtype=jnp.float32) #interconnections
        E = jax.random.uniform(init_keys[1], (num_neurons,num_obs), minval=-init_param_span, maxval=init_param_span, dtype=jnp.float32) #mapping from observations to neurons
        D = jax.random.uniform(init_keys[2], (num_actions,num_neurons), minval=-init_param_span, maxval=init_param_span, dtype=jnp.float32) #readout

        tau = jax.random.uniform(init_keys[3], (num_neurons,), minval=-init_param_span, maxval=init_param_span, dtype=jnp.float32) # time constants for each neuron
        B = jax.random.uniform(init_keys[4], (num_neurons,), minval=-init_param_span, maxval=init_param_span, dtype=jnp.float32) # bias for each neuron


        params = Params(content={'J':J, 'tau':tau, 'E':E, 'B':B, 'D':D})
        return Policy(params=params, state=state, key=key)
    
    @staticmethod
    @jax.jit
    def step_policy(policy:Policy, input:Signal, params:Params):
        dt = params.content['dt']
        action_scale = params.content['action_scale']
        time_constant_scale = params.content['time_constant_scale']

        J = policy.params.content['J']
        tau = policy.params.content['tau']
        E = policy.params.content['E']
        B = policy.params.content['B']
        D = policy.params.content['D']
        
        Z = policy.state.content['Z']

        # get the input
        obs = input.content['obs']

        # step the policy
        z_dot = jnp.tanh(jnp.matmul(J, Z) + jnp.matmul(E, obs) + B) - Z
        z_dot = jnp.multiply(z_dot, time_constant_scale*jax.nn.sigmoid(tau))
        
        new_Z = Z + dt*z_dot # simple euler integration
        
        # ablations for different seeds
        #new_Z = new_Z.at[special_neuron_seed_7[0]].set(1.0)
        #new_Z = new_Z.at[special_neuron_seed_11[1]].set(1.0)
        #new_Z = new_Z.at[special_neuron_seed_11[0]].set(1.0)
        #new_Z = new_Z.at[special_neuron_seed_5[0]].set(1.0)
        
        readout = jnp.matmul(D, new_Z)
        # 0-> speed(sigmoid) 1-> angular speed (tanh)
        actions = action_scale * jnp.array([jax.nn.sigmoid(readout[0]), jnp.tanh(readout[1])])

        new_policy_state = State(content={'Z': new_Z, 'action': actions})
        new_policy = policy.replace(state = new_policy_state)
        
        return new_policy
    
    @staticmethod
    @jax.jit
    def set_policy(policy:Policy, set_params:Params):
        """
        Set the parameters of the policy to the given parameters
        Args:
            - policy: The policy to set
            - set_params: The parameters to set the policy to
        Returns:
            The updated policy
        """
        J = set_params.content['J']
        tau = set_params.content['tau']
        E = set_params.content['E']
        B = set_params.content['B']
        D = set_params.content['D']
        new_policy_params = Params(content={'J':J, 'tau':tau, 'E':E, 'B':B, 'D':D})
        return policy.replace(params = new_policy_params)


@struct.dataclass
class Forager(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
        policy = params.content['policy']
        
        x_max = params.content["x_max"] # max x in spawning arena
        y_max = params.content["y_max"] # max y in spawning arena
        energy_begin_max = params.content["energy_begin_max"] # max energy at the beginning
        radius = params.content["radius"] # radius of the forager

        key, *subkeys = random.split(key, 5)
        
        x = jax.lax.cond(id==0,
                         lambda _: jnp.array([0.0]),
                         lambda _: 0.4*jnp.array([RAY_MAX_LENGTH + FORAGER_RADIUS - 10e-4]), # ~ 50 units away from center
                         None)
        y = jnp.array([0.0])
        ang = jnp.array([0.0])
        
        
        x_dot = jnp.zeros((1,), dtype=jnp.float32)  # initial x velocity
        y_dot = jnp.zeros((1,), dtype=jnp.float32)  # initial y velocity
        ang_dot = jnp.zeros((1,), dtype=jnp.float32)  # angular velocity

        
        energy = jax.lax.cond(id==0,
                              lambda _: jnp.array([energy_begin_max]),
                              lambda _: jnp.array([CLAMPED_FORAGER_ENERGY_BEGIN_MAX]),
                              None)
        
        fitness = jnp.array([0.0])


        state_content = { "x": x, "y": y, "x_dot": x_dot, "y_dot": y_dot, "ang": ang, "ang_dot": ang_dot, "energy": energy , "fitness": fitness }
        state = State(content = state_content)

        params_content = {"radius": radius, "x_max": x_max, "y_max": y_max, "energy_begin_max": energy_begin_max}
        params = Params(content = params_content)

        return Forager(id = id, active_state = active_state, age = 0.0, agent_type = type, params = params, state = state, policy = policy, key = key)

    @staticmethod
    def step_agent(agent, input, step_params):
        obs_content = input.content["obs"]

        energy_in = input.content["energy_in"]
        is_in_sum = input.content["is_in_sum"]
        
        energy = agent.state.content["energy"]
        x = agent.state.content["x"]
        y = agent.state.content["y"]
        ang = agent.state.content["ang"]
        x_dot = agent.state.content["x_dot"]
        y_dot = agent.state.content["y_dot"]
        ang_dot = agent.state.content["ang_dot"]

        obs_content = {'obs': jnp.concatenate((obs_content, 
                                               energy, 
                                               jnp.array([energy_in]),
                                               jnp.array([is_in_sum]), 
                                               x_dot , y_dot, ang_dot), axis=0)} #3*RAY_RESOLUTION + 6
        obs = Signal(content=obs_content)

        new_policy = CTRNN.step_policy(agent.policy, obs, step_params)  #step_params contains dt, action_scale, time_constant_scale
        
        dt = step_params.content["dt"]
        damping = step_params.content["damping"]
        metabolic_cost_speed = step_params.content["metabolic_cost_speed"]
        metabolic_cost_angular = step_params.content["metabolic_cost_angular"]
        x_max_arena = step_params.content["x_max_arena"]
        y_max_arena = step_params.content["y_max_arena"]
        edge_penalty = step_params.content["edge_penalty"]

        
        
        fitness = agent.state.content["fitness"]

        action = new_policy.state.content['action']
        key, *noise_keys = random.split(agent.key, 3)
        speed =(LINEAR_ACTION_SCALE*action[0]+LINEAR_ACTION_OFFSET)*(1 + NOISE_SCALE*jax.random.normal(noise_keys[0], ())) # linear speed action
        ang_speed = action[1]*(1 + NOISE_SCALE*jax.random.normal(noise_keys[1], ()))  # angular speed action

        x_new = jnp.clip(x + dt*x_dot, -x_max_arena, x_max_arena)  # clip to arena bounds
        y_new = jnp.clip(y + dt*y_dot, -y_max_arena, y_max_arena)  # clip to arena bounds
        ang_new = jnp.mod(ang + dt*ang_dot + jnp.pi, 2*jnp.pi) - jnp.pi  # wrap angle to [-pi, pi]
    
        x_dot_new = speed*jnp.cos(ang) - dt*x_dot*damping
        y_dot_new = speed*jnp.sin(ang) - dt*y_dot*damping
        ang_dot_new = ang_speed - dt*ang_dot*damping

        metabolic_cost = metabolic_cost_speed * jnp.abs(speed)/(ACTION_SCALE) + metabolic_cost_angular * jnp.abs(ang_speed)/(ACTION_SCALE) + BASIC_METABOLIC_COST # scaled by action scales to make it invariant to action scale changes
        energy_new = energy  #+ energy_in - metabolic_cost

        edge_penalty = jax.lax.cond(jnp.logical_or(jnp.abs(x_new[0]) >= 0.9*x_max_arena, jnp.abs(y_new[0]) >= 0.9*y_max_arena),
                                           lambda _: edge_penalty,
                                           lambda _: 0.0, None)


        #fitness_new = fitness + delta_fitness - edge_penalty
        fitness_new = energy_new - edge_penalty  # using energy as fitness to see if that affects the internal state dependency of the emerged behavior.
        
        new_state_content = { "x": x_new, "y": y_new, "x_dot": x_dot_new, "y_dot": y_dot_new, "ang": ang_new, "ang_dot": ang_dot_new, 
                            "energy": energy_new, "fitness": fitness_new }
        new_state = State(content = new_state_content)

        
        #move agent with id =0 others hold still
        return jax.lax.cond(agent.id==0,
                            lambda _: agent.replace(state = new_state, policy = new_policy, key = key, age = agent.age + dt),
                            lambda _: agent, None)

    @staticmethod
    def reset_agent(agent, reset_params):
        x_max = agent.params.content["x_max"]
        y_max = agent.params.content["y_max"]
        energy_begin_max = agent.params.content["energy_begin_max"]
        key = agent.key

        key, *subkeys = random.split(key, 5)
        x = jax.lax.cond(agent.id==0,
                         lambda _: jnp.array([0.0]),
                         lambda _: 0.4*jnp.array([RAY_MAX_LENGTH + FORAGER_RADIUS - 10e-4]),
                         None)
        y = jnp.array([0.0])
        ang = jnp.array([0.0])

        x_dot = jnp.zeros((1,), dtype=jnp.float32)  # initial x velocity
        y_dot = jnp.zeros((1,), dtype=jnp.float32)  # initial y velocity
        ang_dot = jnp.zeros((1,), dtype=jnp.float32)  # angular velocity
        
        energy = jax.lax.cond(agent.id==0,
                              lambda _: jnp.array([energy_begin_max]),
                              lambda _: jnp.array([CLAMPED_FORAGER_ENERGY_BEGIN_MAX]),
                              None)
        
        fitness = jnp.array([0.0])

        state_content = { "x": x, "y": y, "x_dot": x_dot, "y_dot": y_dot, "ang": ang, 
                        "ang_dot": ang_dot, "energy": energy , "fitness": fitness}
        state = State(content = state_content)

        return agent.replace(state = state, age = 0.0, key = key)




def get_sensor_data(foragers):
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
            
            min_dist = intercepts[min_dist_indx]
            
            sensed_energy, sensed_type = jax.lax.cond(min_dist<ray.length, 
                                                      lambda _: (agent_energies[min_dist_indx], agent_types[min_dist_indx]),
                                                      lambda _: (0.0, 0),  # 0 for empty, 1 for forager, 2 for patch
                                                      None)
            
            sensed_type_value = type_sensor_values[sensed_type]  # get the type sensor value based on the sensed type

            sensed_energy_channels = sensed_energy*sensed_type_value  # energy is only sensed if the type is forager or patch

            return jnp.concatenate((jnp.array([min_dist]), sensed_energy_channels))
        
        return jax.vmap(for_each_ray)(rays).reshape(-1)
    return jax.vmap(for_each_forager)(foragers)

jit_get_sensor_data = jax.jit(get_sensor_data)

    


def agent_interactions(foragers:Forager):
    
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
        is_in_forager = jnp.sum(is_near)  # how many other foragers the forager is in

        dist = jnp.where(ids_foragers != id_forager, dist, MAX_SPAWN_X)
        nn_dist = jnp.min(dist)

        return is_in_forager, nn_dist

    is_in, nn_dists = jax.vmap(forager_forager_interaction, in_axes=(0, None))(foragers, foragers)

    is_in = is_in.reshape(-1)
    return is_in, nn_dists

jit_agent_interactions = jax.jit(agent_interactions)


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
    is_in_sum, NN_dist = jit_agent_interactions(forager_set.agents)
    avg_NN_dist = jnp.mean(NN_dist)
    sensor_data = jit_get_sensor_data(forager_set.agents)

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
    forager_0_zs = render_data.content['forager_zs']

    jnp.save(TEST_RENDER_PATH + "rendering_foragers_xs.npy", foragers_xs)
    jnp.save(TEST_RENDER_PATH + "rendering_foragers_ys.npy", foragers_ys)
    jnp.save(TEST_RENDER_PATH + "rendering_foragers_angs.npy", foragers_angs)
    jnp.save(TEST_RENDER_PATH + "rendering_forager_energies.npy", forager_energies)
    jnp.save(TEST_RENDER_PATH + "rendering_forager_avg_NN_dists.npy", forager_avg_NN_dists)
    jnp.save(TEST_RENDER_PATH + "rendering_forager_0_zs.npy", forager_0_zs)

if __name__ == "__main__":
    ES_Params = jnp.load("../../params/seed_" + str(seed) + "/params_list.npy")
    #remove the last  params set as it is ineffective saved because the in the last generation
    ES_Params = ES_Params[:-1,:]
    ES_Params = ES_Params[-1:,:] # just the last set of params
    
    
    key = jnp.load("../../params/seed_" + str(seed) + "/test_key.npy")
    key, swarm_key = jax.random.split(key)
    test_param(ES_Params, swarm_key)