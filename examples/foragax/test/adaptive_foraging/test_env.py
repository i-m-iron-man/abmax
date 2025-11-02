from abmax.structs import *
from abmax.functions import *
import jax.numpy as jnp
import jax.random as random
import jax
from flax import struct
from evosax import CMA_ES

MAX_WORLD_X = 120.0 # max travel distance in x direction
MAX_WORLD_Y = MAX_WORLD_X # max travel distance in y direction


MAX_SPAWN_X = 80.0 # max x in spawning arena
MAX_SPAWN_Y = MAX_SPAWN_X # max y in spawning arena

Dt = 0.1
KEY = random.PRNGKey(11)
DAMPING = 0.1
NOISE_SCALE = 0.05
DATA_PATH = None

# forager params
NUM_FORAGERS = 10 # W&B update
RAY_RESOLUTION = 9 # W&B update
RAY_SPAN = jnp.pi/3 # W&B update
RAY_MAX_LENGTH = 120.0 
FORAGER_RADIUS = 5.0

METABOLIC_COST_ANGULAR = 0.05
METABOLIC_COST_SPEED = METABOLIC_COST_ANGULAR/ FORAGER_RADIUS # making it proportional to radius
BASIC_METABOLIC_COST = 0.02 # idle cost independent of speed

FORAGER_ENERGY_BEGIN_MAX = 10.0
FORAGER_AGENT_TYPE = 1

# controller params
NUM_OBS = 3*RAY_RESOLUTION + 6 # 5 = energy + energy_in + x_dot + y_dot + theta_dot + is_in_sum +each_ray[distance, energy_forager, energy_patch]
NUM_NEURONS = 40 # W&B update
NUM_ACTIONS = 2
ACTION_SCALE = 1.0
LINEAR_ACTION_SCALE = ACTION_SCALE * FORAGER_RADIUS # scale for linear action compared to angular action
LINEAR_ACTION_OFFSET = 0.0 # offset for linear action to make the forager move forward by default

NUM_ES_PARAMS = NUM_NEURONS * (NUM_NEURONS + NUM_OBS + NUM_ACTIONS + 2) # 2 for bias and time constants
INIT_PARAM_SPAN = 5.0 # W&B update
TIME_CONSTANT_SCALE = 10.0

# patch params
PATCH_AGENT_TYPE = 2
NUM_PATCH = 10#150 # env lumpiness
ENERGY_MAX = 10.0 # env lumpiness
PATCH_RADIUS = 5.0 # env lumpiness
GROWING_RATE = 0.1 # env lumpiness
EAT_RATE = 0.3 # env lumpiness

# training params
NUM_WORLDS = 8 # W&B update
NUM_GENERATIONS = 2000 # W&B update
POPULATION_SIZE = NUM_FORAGERS # W&B update
EDGE_PENALTY = 100.0
EP_LEN = 2000 
ELITE_RATIO = 0.3 # W&B update
SIGMA_INIT = 0.1 # W&B update

FITNESS_THRESH_SAVE = -100.0 # threshold for saving render data
FITNESS_THRESH_SAVE_STEP = 30.0 # the amount by which we increase the threshold for saving render data
MAX_EXPECTED_FITNESS = 450.0 # will be useful for adaptive saving step

FORAGING_WORLD_PARAMS = Params(content = { "patch_params":{
    "x_max": MAX_SPAWN_X,
    "y_max": MAX_SPAWN_Y,
    "energy_max": ENERGY_MAX,
    "growth_rate": GROWING_RATE,
    "eat_rate": EAT_RATE,
    "radius": PATCH_RADIUS,
    "agent_type": PATCH_AGENT_TYPE,
    "num_patches": NUM_PATCH
},
"forager_params":{
    "x_max": MAX_SPAWN_X,
    "y_max": MAX_SPAWN_Y,
    "energy_begin_max": FORAGER_ENERGY_BEGIN_MAX,
    "radius": FORAGER_RADIUS,
    "agent_type": FORAGER_AGENT_TYPE,
    "num_foragers": NUM_FORAGERS
},
"policy_params":{
    "num_neurons": NUM_NEURONS,
    "num_obs": NUM_OBS,
    "num_actions": NUM_ACTIONS,
    "init_param_span": INIT_PARAM_SPAN
}})

@struct.dataclass
class Patch(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
        key, *subkeys = random.split(key,4)

        x_max = params.content["x_max"] # max x in spawning arena
        y_max = params.content["y_max"] # max y in spawning arena
        energy_max = params.content["energy_max"] # max energy
        eat_rate = params.content["eat_rate"] # rate at which the agent eats food if present

        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        energy = jnp.array([energy_max])#random.uniform(subkeys[2], shape=(1,), minval=10e-4, maxval=energy_max)
        energy_offer = eat_rate * energy # energy offered to foragers
        
        state_content = { "x": x, "y": y, "energy": energy , "energy_offer": energy_offer } # all shapes are (1,)
        state = State( content = state_content)

        growth_rate = params.content["growth_rate"] # growth rate of the patch
        radius = params.content["radius"] # radius of the patch

        param_content = { "growth_rate": growth_rate, "radius": radius, "eat_rate": eat_rate, "energy_max": energy_max, "x_max": x_max, "y_max": y_max }
        params = Params(content = param_content)

        return Patch(id=id, active_state = active_state, age = 0.0, agent_type = type, params = params, state = state, policy = None, key = key)

    @staticmethod
    def step_agent(agent, input, step_params):
        eat_rate = agent.params.content["eat_rate"]
        growth_rate = agent.params.content["growth_rate"]
        energy_max = agent.params.content["energy_max"]
        
        dt = step_params.content["dt"]
        is_energy_out = input.content["is_energy_out"] # T/F
        energy_offer = agent.state.content["energy_offer"]

        energy = agent.state.content["energy"]

        new_energy = energy + energy*growth_rate
        new_energy = new_energy - is_energy_out * energy_offer
        new_energy = jnp.clip(new_energy, 10e-4, energy_max)

        new_energy_offer = eat_rate * new_energy
        new_state_content = { "x": agent.state.content["x"], "y": agent.state.content["y"], "energy": new_energy, "energy_offer": new_energy_offer }
        new_state = State(content = new_state_content)

        return agent.replace(state = new_state, age = agent.age + dt)
    
    @staticmethod
    def reset_agent(agent, reset_params):
        x_max = agent.params.content["x_max"]
        y_max = agent.params.content["y_max"]
        energy_max = agent.params.content["energy_max"]
        eat_rate = agent.params.content["eat_rate"]
        key = agent.key
        
        key, *subkeys = random.split(key, 4)
        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        energy = jnp.array([energy_max])#random.uniform(subkeys[2], shape=(1,), minval=10e-4, maxval=energy_max)
        energy_offer = eat_rate * energy  # energy offered to foragers
        
        state_content = { "x": x, "y": y, "energy": energy, "energy_offer": energy_offer }
        state = State(content = state_content)

        return agent.replace(state = state, age = 0.0, key = key)


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
        readout = jnp.matmul(D, new_Z)
        # 0-> speed(sigmoid) 1-> angular speed (tanh)
        actions = action_scale * jnp.array([jax.nn.sigmoid(readout[0]), jnp.tanh(readout[1])])
        #actions = action_scale * jnp.array([jnp.tanh(readout[0]), jnp.tanh(readout[1])])

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
        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)  # initial angle
        x_dot = jnp.zeros((1,), dtype=jnp.float32)  # initial x velocity
        y_dot = jnp.zeros((1,), dtype=jnp.float32)  # initial y velocity
        ang_dot = jnp.zeros((1,), dtype=jnp.float32)  # angular velocity

        #energy = jnp.array([energy_begin_max])  # initial energy, can be randomised later
        energy = random.uniform(subkeys[3], shape=(1,), minval=0.5*energy_begin_max, maxval=energy_begin_max)
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
        energy_new = energy + energy_in - metabolic_cost

        edge_penalty = jax.lax.cond(jnp.logical_or(jnp.abs(x_new[0]) >= 0.9*x_max_arena, jnp.abs(y_new[0]) >= 0.9*y_max_arena),
                                           lambda _: edge_penalty,
                                           lambda _: 0.0, None)


        #fitness_new = fitness + delta_fitness - edge_penalty
        fitness_new = energy_new - edge_penalty  # using energy as fitness to see if that affects the internal state dependency of the emerged behavior.
        
        new_state_content = { "x": x_new, "y": y_new, "x_dot": x_dot_new, "y_dot": y_dot_new, "ang": ang_new, "ang_dot": ang_dot_new, 
                            "energy": energy_new, "fitness": fitness_new }
        new_state = State(content = new_state_content)

        return agent.replace(state = new_state, policy = new_policy, key = key, age = agent.age + dt)
    
    @staticmethod
    def reset_agent(agent, reset_params):
        x_max = agent.params.content["x_max"]
        y_max = agent.params.content["y_max"]
        energy_begin_max = agent.params.content["energy_begin_max"]
        key = agent.key

        key, *subkeys = random.split(key, 5)
        x = random.uniform(subkeys[0], shape=(1,), minval=-x_max, maxval=x_max)
        y = random.uniform(subkeys[1], shape=(1,), minval=-y_max, maxval=y_max)
        ang = random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)  # initial angle
        x_dot = jnp.zeros((1,), dtype=jnp.float32)  # initial x velocity
        y_dot = jnp.zeros((1,), dtype=jnp.float32)  # initial y velocity
        ang_dot = jnp.zeros((1,), dtype=jnp.float32)  # angular velocity

        #energy = jnp.array([energy_begin_max])
        energy = random.uniform(subkeys[3], shape=(1,), minval=0.5*energy_begin_max, maxval=energy_begin_max)
        fitness = jnp.array([0.0])

        state_content = { "x": x, "y": y, "x_dot": x_dot, "y_dot": y_dot, "ang": ang, 
                        "ang_dot": ang_dot, "energy": energy , "fitness": fitness}
        state = State(content = state_content)

        return agent.replace(state = state, age = 0.0, key = key)

@struct.dataclass
class Point:
    x: jnp.float32
    y: jnp.float32

@struct.dataclass
class Line:
    p1: Point
    p2: Point

@struct.dataclass
class Circle:
    center: Point
    radius: jnp.float32

@struct.dataclass
class Ray:
    origin: Point
    direction: Point # cos, sin
    length: jnp.float32

def generate_rays(agent_pos:Forager, ray_span:jnp.float32, ray_length:jnp.float32):
    """
    Generate rays for the forager agent given the agent position, the span of the rays, the length of the rays and the number of rays
    Args:
        - agent_pos: The position of the agent: (x, y, angle)
        - ray_span: The span of the rays i.e. how wide the forager can see by 2
        - ray_length: The length of the rays
        - RAY_RESOLUTION: The number of rays is global
    Returns:
        The rays generated by the foraer agent at the given position

    """
    x, y, angle = agent_pos

    ray_angles = jnp.linspace(angle - ray_span, angle + ray_span, RAY_RESOLUTION)
    cos_ray_angles = jnp.cos(ray_angles)
    sin_ray_angles = jnp.sin(ray_angles)
    ray_directions = jax.vmap(Point)(cos_ray_angles, sin_ray_angles)
    
    ray_origin = Point(x, y)
    
    rays = jax.vmap(Ray, in_axes=(None, 0, None))(ray_origin, ray_directions, ray_length)
    return rays

jit_generate_rays = jax.jit(generate_rays)

def get_ray_circle_collision(ray:Ray, circle:Circle):
    """
    ray casting algorithm to check for collision between a ray and a circle, adapted from https://www.youtube.com/watch?v=ebzlMOw79Yw&ab_channel=MagellanicMath
    Args:
        - ray: Ray, The ray to check for collision
        - circle: Circle, The circle to check for collision
    Returns:
        The distance of the collision of the ray with the circle along the ray
    """
    circle_center = jnp.reshape(jnp.array([circle.center.x, circle.center.y]), (2,))
    ray_origin = jnp.reshape(jnp.array([ray.origin.x, ray.origin.y]), (2,))
    ray_direction = jnp.reshape(jnp.array([ray.direction.x, ray.direction.y]), (2,))

    s = ray_origin - circle_center
    b = jnp.dot(s, ray_direction)
    c = jnp.dot(s, s) - circle.radius**2
    h = b**2 - c
    h = jax.lax.cond(h < 0, lambda _: -1.0, lambda _: jnp.sqrt(h), None)
    t = jax.lax.cond(h >= 0, lambda _: -b - h, lambda _: ray.length, None)
    t = jax.lax.cond(t < 0, lambda _: ray.length, lambda _: t, None) # this enables the forager to see ouside the circle
    return jnp.minimum(t, ray.length)

jit_get_ray_circle_collision = jax.jit(get_ray_circle_collision)

def agent_interactions(foragers:Forager, patches:Patch):

    def forager_patches_interaction(forager, patches):
        xs_patches = patches.state.content['x']  # an array of shape (num_patches,)
        ys_patches = patches.state.content['y']
        x_forager = forager.state.content['x']  # a 1x1 array
        y_forager = forager.state.content['y']

        dists = jnp.linalg.norm(jnp.stack((xs_patches - x_forager, ys_patches - y_forager), axis=1), axis=1).reshape(-1)  # distances from the forager to the patches
        is_near = jnp.where(dists < patches.params.content['radius'], 1.0, 0.0)  # 1 if the forager is near the patch, 0 otherwise
        
        #energy_offers = jnp.multiply(is_near, patches.state.content['energy_offer'].reshape(-1))  # energy offers from the patches
        
        is_in_patch = jnp.sum(is_near)  # how many patches the forager is in
        
        return is_in_patch, is_near#energy_offers
    
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
        return is_in_forager
    
    is_in_patches, is_near_matrix = jax.vmap(forager_patches_interaction, in_axes=(0, None))(foragers, patches)
    is_in_foragers = jax.vmap(forager_forager_interaction, in_axes=(0, None))(foragers, foragers)

    is_in_sum = is_in_patches + is_in_foragers

    #energy_in_foragers = jnp.sum(energy_offers, axis=1).reshape(-1)  # sum the energy offers from the patches for each forager
    is_energy_out_patches = jnp.any(is_near_matrix, axis=0)  # t/f if a patch is being foraged by any forager

    num_foragers_present_per_patch = jnp.maximum(jnp.sum(is_near_matrix, axis=0),1.0)  # number of foragers present at each patch
    # point of maximum with 1.0: to avoid division by zero, and if there is 1 forager, it will just be 1.0
    energy_sharing_matrix = jnp.divide(is_near_matrix, num_foragers_present_per_patch)  # each forager gets an equal share of the patch's energy offer
    energy_in_foragers = jnp.multiply(energy_sharing_matrix, patches.state.content['energy_offer'].reshape(-1))  # energy received by each forager from each patch
    energy_in_foragers = jnp.sum(energy_in_foragers, axis=1).reshape(-1)  # sum the energy offers from the patches for each forager
    return is_in_sum, is_energy_out_patches, energy_in_foragers#energy_in_foragers, energy_out_patches

jit_agent_interactions = jax.jit(agent_interactions)


def get_sensor_data(foragers:Forager, patches:Patch):
    agent_xs = jnp.concatenate((foragers.state.content['x'].reshape(-1), patches.state.content['x'].reshape(-1)))
    agent_ys = jnp.concatenate((foragers.state.content['y'].reshape(-1), patches.state.content['y'].reshape(-1)))
    agent_rads = jnp.concatenate((foragers.params.content['radius'].reshape(-1), patches.params.content['radius'].reshape(-1)))
    agent_energies = jnp.concatenate((foragers.state.content['energy'].reshape(-1), patches.state.content['energy'].reshape(-1)))
    agent_types = jnp.concatenate((foragers.agent_type, patches.agent_type))

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

def set_CMAES_params(CMAES_params:Params, foragers:Forager): 
    """
    copy the CMAES_params to the foragers while manipulating the shape of the parameters
    Args:
        - CMAES_params: The parameters to set with shape (NUM_FORAGERS, NUM_ES_PARAMS)
        - foragers: The foragers to set the parameters to

    Returns:
        The updated foragers
    """
    J = CMAES_params[:,:NUM_NEURONS*NUM_NEURONS].reshape((-1, NUM_NEURONS, NUM_NEURONS)) # J: (NUM_FORAGERS, NUM_NEURONS, NUM_NEURONS)
    last_index = NUM_NEURONS*NUM_NEURONS

    tau = CMAES_params[:,last_index:last_index + NUM_NEURONS].reshape((-1, NUM_NEURONS)) # tau: (NUM_FORAGERS, NUM_NEURONS,)
    last_index += NUM_NEURONS

    E = CMAES_params[:,last_index:last_index + NUM_NEURONS*NUM_OBS].reshape((-1, NUM_NEURONS, NUM_OBS)) # E: (NUM_FORAGERS, NUM_NEURONS, NUM_OBS)
    last_index += NUM_NEURONS*NUM_OBS

    B = CMAES_params[:,last_index:last_index + NUM_NEURONS].reshape((-1, NUM_NEURONS)) # B: (NUM_FORAGERS, NUM_NEURONS)
    last_index += NUM_NEURONS

    D = CMAES_params[:,last_index:last_index + NUM_NEURONS*NUM_ACTIONS].reshape((-1, NUM_ACTIONS, NUM_NEURONS)) # D: (NUM_FORAGERS, NUM_ACTIONS, NUM_NEURONS)

    policy_params = Params(content={'J':J, 'tau':tau, 'E':E, 'B':B, 'D':D})
    new_policies = jax.vmap(CTRNN.set_policy)(foragers.policy, policy_params)
    return foragers.replace(policy = new_policies)
jit_set_CMAES_params = jax.jit(set_CMAES_params)

@struct.dataclass
class Foraging_world():
    forager_set: Set
    patch_set: Set

    @staticmethod
    def create_foraging_world(params, key):
        patch_params = params.content['patch_params']
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
        
        key, patch_key = random.split(key, 2)

        num_patches = patch_params['num_patches']
        x_max_array = jnp.tile(jnp.array([patch_params['x_max']]), (num_patches,))
        y_max_array = jnp.tile(jnp.array([patch_params['y_max']]), (num_patches,))
        energy_max_array = jnp.tile(jnp.array([patch_params['energy_max']]), (num_patches,))
        eat_rate_array = jnp.tile(jnp.array([patch_params['eat_rate']]), (num_patches,))
        growth_rate_array = jnp.tile(jnp.array([patch_params['growth_rate']]), (num_patches,))
        radius_array = jnp.tile(jnp.array([patch_params['radius']]), (num_patches,))

        patch_create_params = Params(content={ 'x_max': x_max_array,
                                                'y_max': y_max_array,
                                                'energy_max': energy_max_array,
                                                'eat_rate': eat_rate_array,
                                                'growth_rate': growth_rate_array,
                                                'radius': radius_array})

        patches = create_agents(agent=Patch, params=patch_create_params, num_agents=num_patches, num_active_agents=num_patches, 
                                agent_type = patch_params['agent_type'], key=patch_key)
        patch_set = Set(num_agents=num_patches, num_active_agents=num_patches, agents=patches, id=1, set_type=patch_params['agent_type'], 
                        params=None, state=None, policy=None, key=None)

        return Foraging_world(forager_set=forager_set, patch_set=patch_set)

def step_world(foraging_world, _t):
    forager_set = foraging_world.forager_set
    patch_set = foraging_world.patch_set
        
    is_in_sum, is_energy_out_patches, energy_in_foragers = jit_agent_interactions(forager_set.agents, patch_set.agents)
    sensor_data = jit_get_sensor_data(forager_set.agents, patch_set.agents)

    patch_step_input = Signal(content={'is_energy_out': is_energy_out_patches})
    patch_step_params = Params(content={'dt': Dt})
    patch_set = jit_step_agents(Patch.step_agent, patch_step_params, patch_step_input, patch_set)

    forager_step_input = Signal(content={'obs': sensor_data, 'energy_in': energy_in_foragers, 'is_in_sum': is_in_sum})
    forager_step_params = Params(content={'dt': Dt, 
                                            'damping': DAMPING, 
                                            'metabolic_cost_speed': METABOLIC_COST_SPEED,
                                            'metabolic_cost_angular': METABOLIC_COST_ANGULAR,
                                            'x_max_arena': MAX_WORLD_X, 
                                            'y_max_arena': MAX_WORLD_Y,
                                            'edge_penalty': EDGE_PENALTY,
                                            'action_scale': ACTION_SCALE,
                                            'time_constant_scale': TIME_CONSTANT_SCALE
                                            })
        
    forager_set = jit_step_agents(Forager.step_agent, forager_step_params, forager_step_input, forager_set)

    render_data = Signal(content={
        'forager_xs': forager_set.agents.state.content['x'].reshape(-1, 1),
        'forager_ys': forager_set.agents.state.content['y'].reshape(-1, 1),
        'forager_angs': forager_set.agents.state.content['ang'].reshape(-1, 1),
        'forager_energies': forager_set.agents.state.content['energy'].reshape(-1, 1),
        'patch_energies': patch_set.agents.state.content['energy'].reshape(-1, 1)
    })

    return foraging_world.replace(patch_set=patch_set, forager_set=forager_set), render_data

jit_step = jax.jit(step_world)


def reset_world(foraging_world):
    foraging_set_agents = foraging_world.forager_set.agents
    patch_set_agents = foraging_world.patch_set.agents

    foraging_set_agents = jax.vmap(Forager.reset_agent)(foraging_set_agents, None)
    patch_set_agents = jax.vmap(Patch.reset_agent)(patch_set_agents, None)

    forager_set = foraging_world.forager_set.replace(agents=foraging_set_agents)
    patch_set = foraging_world.patch_set.replace(agents=patch_set_agents)

    return foraging_world.replace(forager_set=forager_set, patch_set=patch_set)

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
    render_data = Signal(content={
        'forager_xs': render_data.content['forager_xs'], # shape: (num_foragers, EP_LEN)
        'forager_ys': render_data.content['forager_ys'], # shape: (num_foragers, EP_LEN)
        'forager_angs': render_data.content['forager_angs'], # shape: (num_foragers, EP_LEN),
        'forager_energies': render_data.content['forager_energies'], # shape: (num_foragers, EP_LEN)
        'patch_xs': foraging_world.patch_set.agents.state.content['x'].reshape(-1,), # shape: (num_patches,)
        'patch_ys': foraging_world.patch_set.agents.state.content['y'].reshape(-1,), # shape: (num_patches,)
        'patch_energies': render_data.content['patch_energies'] # shape: (num_patches, EP_LEN)
    })
    return foraging_world, render_data

jit_run_episode = jax.jit(run_episode)


def get_fitness(CMAES_params, foraging_worlds):
    
    cmaes_foragers = jax.vmap(jit_set_CMAES_params, in_axes=(None, 0))(CMAES_params, foraging_worlds.forager_set.agents)
    cmaes_forager_set = foraging_worlds.forager_set.replace(agents = cmaes_foragers)
    foraging_worlds = foraging_worlds.replace(forager_set = cmaes_forager_set)

    foraging_worlds, render_data = jax.vmap(jit_run_episode)(foraging_worlds)
    fitness = jnp.reshape(jnp.mean(foraging_worlds.forager_set.agents.state.content['fitness'], axis=0), (-1))

    return fitness, foraging_worlds

jit_get_fitness = jax.jit(get_fitness)


def main():
    key, *foraging_world_keys = random.split(KEY, NUM_WORLDS + 1)
    foraging_world_keys = jnp.array(foraging_world_keys)

    foraging_worlds = jax.vmap(Foraging_world.create_foraging_world, in_axes=(None, 0))(FORAGING_WORLD_PARAMS, foraging_world_keys)
    
    key, subkey = random.split(key)
    strategy = CMA_ES(popsize=NUM_FORAGERS, num_dims=NUM_ES_PARAMS, elite_ratio=ELITE_RATIO, sigma_init= SIGMA_INIT)
    es_params = strategy.default_params
    state = strategy.initialize(subkey, es_params)
    
    mean_fitness_list = []
    max_fitness_list = []
    min_fitness_list = []
    saved_fitness_list = []
    param_list = []
    fitness_thresh_save = FITNESS_THRESH_SAVE
    fitness_thresh_save_step = FITNESS_THRESH_SAVE_STEP

    for generation in range(NUM_GENERATIONS):
        key, gen_key = jax.random.split(key, 2)
        x, state = strategy.ask(gen_key, state, es_params)
        fitness, foraging_worlds = jit_get_fitness(x, foraging_worlds)
        state = strategy.tell(x, -1*fitness, state, es_params)

        mean_fitness = jnp.mean(fitness)
        best_fitness = jnp.max(fitness)
        worst_fitness = jnp.min(fitness)

        mean_fitness_list.append(mean_fitness)
        max_fitness_list.append(best_fitness)
        min_fitness_list.append(worst_fitness)

        if mean_fitness > fitness_thresh_save or generation == NUM_GENERATIONS - 1:
            fitness_thresh_save += fitness_thresh_save_step
            if fitness_thresh_save > 0.0: #apply the scaling of save step using MAX_EXPECTED_FITNESS
                fitness_thresh_save_step = FITNESS_THRESH_SAVE_STEP*(1-(fitness_thresh_save/MAX_EXPECTED_FITNESS))
            saved_fitness_list.append(mean_fitness)
            param_list.append(state.mean)
        
        print('Generation:', generation, 'Mean Fitness:', mean_fitness, 'Best Fitness:', best_fitness, 'Worst Fitness:', worst_fitness)
    
    params_list = jnp.array(param_list)
    saved_fitness_list = jnp.array(saved_fitness_list)
    mean_fitness_list = jnp.array(mean_fitness_list)
    max_fitness_list = jnp.array(max_fitness_list)
    min_fitness_list = jnp.array(min_fitness_list)
    jnp.save(DATA_PATH+ 'params_list.npy', params_list)
    jnp.save(DATA_PATH + 'saved_fitness_list.npy', saved_fitness_list)
    jnp.save(DATA_PATH + 'mean_fitness_list.npy', mean_fitness_list)
    jnp.save(DATA_PATH + 'max_fitness_list.npy', max_fitness_list)
    jnp.save(DATA_PATH + 'min_fitness_list.npy', min_fitness_list)
    jnp.save(DATA_PATH + 'test_key.npy', jnp.array(key))

if __name__ == "__main__":
    main()
    print("Foraging world simulation completed.")


