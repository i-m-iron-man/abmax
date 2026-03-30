'''
In interest of time:
we are not adding the idea of cool down period for child, for interaction.
as that will have a massive refactoring effect on the code, and we want to first see the results without it. We can add it later if needed.
'''


from abmax.structs import *
from abmax.functions import *
import jax.numpy as jnp
import jax.random as random
import jax
from flax import struct
from evosax import CMA_ES

from sim_params import *
from neuron_policy import *

@struct.dataclass
class Boid(Agent):

    @staticmethod
    def create_agent(type, params, id, active_state, key):

        policy = params.content['policy']
        # intialization
        #key, *subkeys = random.split(key, 5)
        #energy = jax.lax.cond(active_state, lambda _: random.uniform(subkeys[3], shape=(1,), minval=0.2*MAX_ENERGY, maxval=0.5*MAX_ENERGY), lambda _: jnp.zeros((1,), dtype=jnp.float32), None) # if active, initialize with random energy, else initialize with 0 energy 

        #x = jax.lax.cond(active_state, lambda _: random.uniform(subkeys[0], shape=(1,), minval=-MAX_SPAWN_X, maxval=MAX_SPAWN_X), lambda _: jnp.array([MAX_WORLD_X + 2*RAY_LENGTH]), None) # if active, spawn randomly, else spawn at MAX_WORLD_X + 2*RAY_LENGTH to avoid spawning in the world
        #y = jax.lax.cond(active_state, lambda _: random.uniform(subkeys[1], shape=(1,), minval=-MAX_SPAWN_Y, maxval=MAX_SPAWN_Y), lambda _: jnp.array([MAX_WORLD_Y + 2*RAY_LENGTH]), None) # if active, spawn randomly, else spawn at MAX_WORLD_Y + 2*RAY_LENGTH to avoid spawning in the world

        state_content = { "x": jnp.zeros((1,), dtype=jnp.float32), #x,
                          "y": jnp.zeros((1,), dtype=jnp.float32), #y,
                          "ang": jnp.zeros((1,), dtype=jnp.float32), #random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi),
                          
                          "x_dot": jnp.zeros((1,), dtype=jnp.float32),  # initial x velocity
                          "y_dot": jnp.zeros((1,), dtype=jnp.float32),  # initial y velocity
                          "ang_dot": jnp.zeros((1,), dtype=jnp.float32),  # initial angular velocity
                          
                          "l_acc": jnp.zeros((1,), dtype=jnp.float32),  # initial linear acceleration
                          "a_acc": jnp.zeros((1,), dtype=jnp.float32),  # initial angular acceleration
                          
                          "movement": jnp.zeros((1,), dtype=jnp.float32), # initial average movement of the agent

                          "energy": jnp.zeros((1,), dtype=jnp.float32), #energy,
                          "bar_energy": jnp.zeros((1,), dtype=jnp.float32), # initial bar energy is same as actual energy

                          "grazing_energy": jnp.zeros((1,), dtype=jnp.float32),
                          "bar_grazing_energy": jnp.zeros((1,), dtype=jnp.float32),
                          "grazing_energy_sum": jnp.zeros((1,), dtype=jnp.float32), # sum of grazing energy obtained from all neighbors in the current step, used for normalization in the selector network
                          
                          "exchange_energy": jnp.zeros((1,), dtype=jnp.float32),
                          "bar_exchange_energy": jnp.zeros((1,), dtype=jnp.float32),
                          "bar_energy_taken": jnp.zeros((1,), dtype=jnp.float32),
                          "abs_energy_exchange_sum": jnp.zeros((1,), dtype=jnp.float32), # absolute sum of energy exchanged with neighbors in the current step, used for normalization in the selector network

                          "metabolic_cost": jnp.zeros((1,), dtype=jnp.float32),
                          "bar_metabolic_cost": jnp.zeros((1,), dtype=jnp.float32),
                          "metabolic_cost_sum": jnp.zeros((1,), dtype=jnp.float32), # sum of metabolic cost of all neighbors in the current step, used for normalization in the selector network

                          "time_death": jnp.zeros((1,), dtype=jnp.float32), # time below death energy threshold
                          "time_reproduction": jnp.zeros((1,), dtype=jnp.float32), # time above reproduction energy threshold

                          "reproduce_flag": False, # flag to indicate if the agent is ready to reproduce
                          "death_flag": False, # flag to indicate if the agent is ready to die

                          "role": jnp.array([0.0]), # false for grazers, true for exchangers
                          "num_switches": jnp.zeros((1,), dtype=jnp.float32)
                        } 
        
        state = State(content = state_content)

        param_state = { 
                        "immortal_flag": active_state # immortal agents cannot die, but can reproduce. This is used to maintain a minimum population size in the environment
                       }
        params = Params(content = param_state)

        
        return Boid(id = id, active_state = active_state, age = 0.0, agent_type = type, params = params, state = state, policy = policy, key = key)
    
    @staticmethod
    def reset_agent(agent, reset_params):
        new_policy = CTRNN.reset_policy(agent.policy)
        key = agent.key
        active_state = agent.params.content['immortal_flag'] # if immortal, keep active, else set to false (dead)

        key, *subkeys = random.split(key, 5)
        energy = jax.lax.cond(active_state, lambda _: random.uniform(subkeys[3], shape=(1,), minval=0.2*MAX_ENERGY, maxval=0.5*MAX_ENERGY), lambda _: jnp.zeros((1,), dtype=jnp.float32), None) # if active, initialize with random energy, else initialize with 0 energy 
        x = jax.lax.cond(active_state, lambda _: random.uniform(subkeys[0], shape=(1,), minval=-MAX_SPAWN_X, maxval=MAX_SPAWN_X), lambda _: jnp.array([MAX_WORLD_X + 2*RAY_LENGTH]), None) # if active, spawn randomly, else spawn at MAX_WORLD_X + 2*RAY_LENGTH to avoid spawning in the world
        y = jax.lax.cond(active_state, lambda _: random.uniform(subkeys[1], shape=(1,), minval=-MAX_SPAWN_Y, maxval=MAX_SPAWN_Y), lambda _: jnp.array([MAX_WORLD_Y + 2*RAY_LENGTH]), None) # if active, spawn randomly, else spawn at MAX_WORLD_Y + 2*RAY_LENGTH to avoid spawning in the world
        role = jax.lax.cond(active_state, lambda _: jnp.array([1.0]), lambda _: jnp.array([0.0]), None) # if active,assignd default role, else assign no role, 0.0-> inactive, 1.0-> active/sunoptimal, 2.0->active/grazer, 3.0->active/exchanger
        state_content = { "x": x,
                          "y": y,
                          "ang": random.uniform(subkeys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi),
                          
                          "x_dot": jnp.zeros((1,), dtype=jnp.float32),  # initial x velocity
                          "y_dot": jnp.zeros((1,), dtype=jnp.float32),  # initial y velocity
                          "ang_dot": jnp.zeros((1,), dtype=jnp.float32),  # initial angular velocity
                          
                          "l_acc": jnp.zeros((1,), dtype=jnp.float32),  # initial linear acceleration
                          "a_acc": jnp.zeros((1,), dtype=jnp.float32),  # initial angular acceleration
                          
                          "movement": jnp.zeros((1,), dtype=jnp.float32), # initial average movement of the agent

                          "energy": energy,
                          "bar_energy": energy, # initial bar energy is same as actual energy
                          
                          "grazing_energy": jnp.zeros((1,), dtype=jnp.float32),
                          "bar_grazing_energy": jnp.zeros((1,), dtype=jnp.float32),
                          "grazing_energy_sum": jnp.zeros((1,), dtype=jnp.float32), # sum of grazing energy obtained from all neighbors in the current step, used for normalization in the selector network
                          
                          "exchange_energy": jnp.zeros((1,), dtype=jnp.float32),
                          "bar_exchange_energy": jnp.zeros((1,), dtype=jnp.float32),
                          "bar_energy_taken": jnp.zeros((1,), dtype=jnp.float32), # energy taken from neighbors in the current step, used for normalization in the selector network
                          "abs_energy_exchange_sum": jnp.zeros((1,), dtype=jnp.float32), # absolute sum of energy exchanged with neighbors in the current step, used for normalization in the selector network

                          "metabolic_cost": jnp.zeros((1,), dtype=jnp.float32),
                          "bar_metabolic_cost": jnp.zeros((1,), dtype=jnp.float32),
                          "metabolic_cost_sum": jnp.zeros((1,), dtype=jnp.float32), # sum of metabolic cost of all neighbors in the current step, used for normalization in the selector network

                          "time_death": jnp.zeros((1,), dtype=jnp.float32), # time below death energy threshold
                          "time_reproduction": jnp.zeros((1,), dtype=jnp.float32), # time above reproduction energy threshold

                          "reproduce_flag": False, # flag to indicate if the agent is ready to reproduce
                          "death_flag": False, # flag to indicate if the agent is ready to die

                          "role": role, # false for grazers, true for exchangers
                          "num_switches": jnp.zeros((1,), dtype=jnp.float32)
                        } 
        
        new_state = State(content = state_content)
        return agent.replace(state = new_state, policy = new_policy, key = key, age = 0.0, active_state = active_state) # if immortal, keep active, else set to false (dead)
    

    @staticmethod
    def step_agent(agent, input, step_params):
        #get data
        ext_obs      = input.content['ext_obs'] # observation vector
        exchange_energy = input.content['energy_delta'].reshape(1) # change in energy
        is_in_sum    = input.content['is_in_sum'] # agent in how many bodies?
        avg_movement = step_params.content['avg_movement'] # average movement of neighbors/world

        x =                  agent.state.content['x']
        y =                  agent.state.content['y']
        x_dot =              agent.state.content['x_dot']
        y_dot =              agent.state.content['y_dot']
        ang =                agent.state.content['ang']
        ang_dot =            agent.state.content['ang_dot']
        l_acc =              agent.state.content['l_acc']
        a_acc =              agent.state.content['a_acc']
        movement =           agent.state.content['movement']

        energy =             agent.state.content['energy']
        bar_energy =         agent.state.content['bar_energy']
        #grazing_energy =     agent.state.content['grazing_energy'] calculated now
        bar_grazing_energy = agent.state.content['bar_grazing_energy']
        #exchange_energy =    agent.state.content['exchange_energy'], input from env
        bar_exchange_energy =agent.state.content['bar_exchange_energy']
        #metabolic_cost =     agent.state.content['metabolic_cost']
        bar_metabolic_cost = agent.state.content['bar_metabolic_cost']
        bar_energy_taken =    agent.state.content['bar_energy_taken']

        grazing_energy_sum = agent.state.content['grazing_energy_sum']
        abs_energy_exchange_sum = agent.state.content['abs_energy_exchange_sum']
        metabolic_cost_sum = agent.state.content['metabolic_cost_sum']

        time_death =         agent.state.content['time_death']
        time_reproduction =  agent.state.content['time_reproduction']
        reproduce_flag =     agent.state.content['reproduce_flag']
        death_flag =         agent.state.content['death_flag']

        role =               agent.state.content['role']
        num_switches =       agent.state.content['num_switches']

        # update_energy
        abs_energy_exchange_sum_new = abs_energy_exchange_sum +jnp.maximum(exchange_energy, 0.0) # minimum will be book-keeped else where thus abs() would have made it 2 times
        bar_exchange_energy_new = ((1 - BAR_ENERGY_TIME_CONSTANT) * bar_exchange_energy + BAR_ENERGY_TIME_CONSTANT * exchange_energy)
        bar_energy_taken_new = ((1 - BAR_ENERGY_TIME_CONSTANT) * bar_energy_taken + BAR_ENERGY_TIME_CONSTANT * jnp.maximum(exchange_energy, 0.0)) # only consider energy taken for the agent, not energy given

        grazing_energy_new = GRAZING_COEFFICIENT * (avg_movement - movement) # energy gained from grazing, proportional to how much the agent is moving compared to the average movement of neighbors/world
        grazing_energy_new = jnp.clip(grazing_energy_new, 0.0, MAX_GRAZING_ENERGY) # clip grazing energy to max grazing energy
        bar_grazing_energy_new = ((1 - BAR_ENERGY_TIME_CONSTANT) * bar_grazing_energy + BAR_ENERGY_TIME_CONSTANT * grazing_energy_new)
        grazing_energy_sum_new = grazing_energy_sum + grazing_energy_new

        metabolic_cost_new = METABOLIC_COST_LINEAR*jnp.abs(l_acc) + METABOLIC_COST_ANGULAR*jnp.abs(a_acc) + BASIC_METABOLIC_COST*energy
        bar_metabolic_cost_new = ((1 - BAR_ENERGY_TIME_CONSTANT) * bar_metabolic_cost + BAR_ENERGY_TIME_CONSTANT * metabolic_cost_new)
        metabolic_cost_sum_new = metabolic_cost_sum + metabolic_cost_new

        energy_new = energy + exchange_energy + grazing_energy_new - metabolic_cost_new # update energy with energy delta and grazing energy, subtract metabolic cost
        energy_new = jnp.clip(energy_new, 0.0, MAX_ENERGY) # clip energy to max energy
        bar_energy_new = ((1 - BAR_ENERGY_TIME_CONSTANT) * bar_energy + BAR_ENERGY_TIME_CONSTANT * energy_new)

        # update reproduction and death timers and flags
        time_death_new = jax.lax.cond(energy_new[0] < DEATH_ENERGY_THRESHOLD, lambda _: time_death + DT, 
                                    lambda _ : jnp.zeros((1,), dtype=jnp.float32), None)
        death_flag_new = jax.lax.cond(death_flag, lambda _: True, lambda _ : time_death_new[0] >= DEATH_TIME_THRESHOLD, None) # can become false, only from the outside when the agent is reset, otherwise once true it stays true
        
        key, subkey = random.split(agent.key)
        old_age_death = jax.lax.cond(agent.age > OLD_AGE_ONSET,
                                     lambda _ : jax.random.bernoulli(subkey, p = (agent.age - OLD_AGE_ONSET)/(MAX_AGE - OLD_AGE_ONSET)),
                                      lambda _ : False, None) # probability of death increases linearly from 0 to 1 between OLD_AGE_ONSET and MAX_AGE
        death_flag_new = jax.lax.cond(jnp.logical_or(death_flag_new, old_age_death), lambda _: True, lambda _ : False, None) # if already dead by energy, stay dead, else can die by old age
        
        death_flag_new = jax.lax.cond(agent.params.content['immortal_flag'], lambda _: False, lambda _ : death_flag_new, None) # if immortal flag is true, agent cannot die, else it can die based on energy and time below threshold
        
        time_reproduction_new = jax.lax.cond(energy_new[0] > REPRODUCTION_ENERGY_THRESHOLD, lambda _: time_reproduction + DT,
                                            lambda _ : jnp.zeros((1,), dtype=jnp.float32), None)
        reproduce_flag_new = jax.lax.cond(reproduce_flag, lambda _: True, lambda _ : time_reproduction_new[0] >= REPRODUCION_TIME_THRESHOLD, None) # can become false, only from the outside when the agent is reset, otherwise once true it stays

        # update role tag and switches based on selector output
        cond_grazing = jnp.logical_and(bar_grazing_energy_new[0] > bar_energy_taken_new[0], bar_grazing_energy_new[0] > bar_metabolic_cost_new[0]) # if energy from grazing is greater than energy from snatching and metabolic cost, be a grazer
        cond_exchanging = jnp.logical_and(bar_energy_taken_new[0] > bar_grazing_energy_new[0], bar_energy_taken_new[0] > bar_metabolic_cost_new[0]) # if energy from snatching is greater than energy from grazing and metabolic cost, be an exchanger
        role_new = jax.lax.cond(cond_grazing, lambda _: jnp.array([2.0]), lambda _: jnp.array([1.0]), None) # 0.0-> inactive, 1.0-> active/sunoptimal, 2.0->active/grazer, 3.0->active/exchanger
        role_new = jax.lax.cond(cond_exchanging, lambda _: jnp.array([3.0]), lambda _: role_new, None) # if both grazing and exchanging energy are
        
        num_switches_new = jax.lax.cond(role_new != role, # if role has switched, increment num_switches
                                        lambda _: num_switches + 1.0, lambda _: num_switches, None)
        
        # update position
        x_new = jnp.clip(x + DT*x_dot, -MAX_WORLD_X, MAX_WORLD_X)  # clip to arena bounds
        y_new = jnp.clip(y + DT*y_dot, -MAX_WORLD_Y, MAX_WORLD_Y)  # clip to arena bounds
        ang_new = jnp.mod(ang + DT*ang_dot + jnp.pi, 2*jnp.pi) - jnp.pi  # wrap angle to [-pi, pi]

        x_dot_new = x_dot + (l_acc*jnp.cos(ang) - x_dot*DAMPING)*DT
        x_dot_new = jnp.clip(x_dot_new, -MAX_VEL, MAX_VEL) # clip velocity to max velocity
        y_dot_new = y_dot + (l_acc*jnp.sin(ang) - y_dot*DAMPING)*DT
        y_dot_new = jnp.clip(y_dot_new, -MAX_VEL, MAX_VEL) # clip velocity to max velocity
        
        ang_dot_new = ang_dot + (a_acc - ang_dot*DAMPING)*DT
        ang_dot_new = jnp.clip(ang_dot_new, -MAX_ANG_VEL, MAX_ANG_VEL) # clip angular velocity to max angular velocity

        movement_new = (1-MOVEMENT_TIME_CONSTANT)*movement + MOVEMENT_TIME_CONSTANT*(jnp.linalg.norm(jnp.stack((x_dot_new, y_dot_new)))+ abs(ang_dot_new[0])) # update movement with smoothing

        # update the policy and actions
        obs_neurons = jnp.concatenate([ext_obs, x_dot_new, y_dot_new, ang_dot_new, jnp.array([is_in_sum]), 
                                        avg_movement, movement_new, energy_new, grazing_energy_new, exchange_energy, metabolic_cost_new]) # concatenate external observation with internal state information for neurons


        obs_content = { "obs": obs_neurons}
        obs = Signal(content= obs_content)

        new_policy = CTRNN.step_policy(agent.policy, obs, step_params)

        key, *noise_keys = random.split(agent.key, 3)
        action = new_policy.state.content['action']

        l_acc_noise = (1.0 + NOISE_SCALE_ACC*jax.random.truncated_normal(noise_keys[0], lower=-1.0, upper=1.0, shape=()))
        l_acc_scale = LINEAR_ACTION_SCALE*l_acc_noise
        l_acc_new = action[0]*l_acc_scale
        l_acc_new = l_acc_new.reshape(1,)

        a_acc_noise = (1.0 + NOISE_SCALE_ACC*jax.random.truncated_normal(noise_keys[1], lower=-1.0, upper=1.0, shape=()))
        a_acc_scale = ACTION_SCALE*a_acc_noise
        a_acc_new = action[1]*a_acc_scale
        a_acc_new = a_acc_new.reshape(1,)

        state_content = {   "x": x_new,
                            "y": y_new,
                            "ang": ang_new,
                            "x_dot": x_dot_new,
                            "y_dot": y_dot_new,
                            "ang_dot": ang_dot_new,
                            "l_acc": l_acc_new,
                            "a_acc": a_acc_new,
                            "movement": movement_new,
                            "energy": energy_new,
                            "bar_energy": bar_energy_new,
                            "grazing_energy": grazing_energy_new,
                            "bar_grazing_energy": bar_grazing_energy_new,
                            "grazing_energy_sum": grazing_energy_sum_new,
                            "exchange_energy": exchange_energy,
                            "bar_exchange_energy": bar_exchange_energy_new,
                            "bar_energy_taken": bar_energy_taken_new,
                            "abs_energy_exchange_sum": abs_energy_exchange_sum_new,
                            "metabolic_cost": metabolic_cost_new,
                            "bar_metabolic_cost": bar_metabolic_cost_new,
                            "metabolic_cost_sum": metabolic_cost_sum_new,
                            "time_death": time_death_new,
                            "time_reproduction": time_reproduction_new,
                            "reproduce_flag": reproduce_flag_new,
                            "death_flag": death_flag_new,
                            "role": role_new,
                            "num_switches": num_switches_new
                        }
        new_state = State(content = state_content)
        new_agent = jax.lax.cond(agent.active_state, lambda _: agent.replace(state = new_state, policy = new_policy, key = key, age = agent.age + 1.0), 
                                 lambda _: agent, None) # only update if agent is active, else keep it the same (it will be reset later when it becomes active)
        return new_agent
    
    



    @staticmethod
    def add_agent(agent, add_params):
        agent_to_copy = add_params.content['agent_to_copy']
        child_J = add_params.content['child_J']

        # update state
        key, *sub_keys = random.split(agent.key, 4)
        x = agent_to_copy.state.content['x'] + random.uniform(sub_keys[0], shape=(1,), minval=-OFFSPRING_SPREAD, maxval=OFFSPRING_SPREAD)
        y = agent_to_copy.state.content['y'] + random.uniform(sub_keys[1], shape=(1,), minval=-OFFSPRING_SPREAD, maxval=OFFSPRING_SPREAD)
        ang = random.uniform(sub_keys[2], shape=(1,), minval=-jnp.pi, maxval=jnp.pi)
        energy = agent_to_copy.state.content['energy']/2.0 # split energy between parent and offspring
        bar_energy = energy # initial bar energy is same as actual energy
        
        state_content = { "x": x,
                          "y": y,
                          "ang": ang,
                          
                          "x_dot": jnp.zeros((1,), dtype=jnp.float32),  # initial x velocity
                          "y_dot": jnp.zeros((1,), dtype=jnp.float32),  # initial y velocity
                          "ang_dot": jnp.zeros((1,), dtype=jnp.float32),  # initial angular velocity
                          
                          "l_acc": jnp.zeros((1,), dtype=jnp.float32),  # initial linear acceleration
                          "a_acc": jnp.zeros((1,), dtype=jnp.float32),  # initial angular acceleration
                          
                          "movement": jnp.zeros((1,), dtype=jnp.float32), # initial average movement of the agent

                          "energy": energy,
                          "bar_energy": bar_energy, # initial bar energy is same as actual energy
                          
                          "grazing_energy": jnp.zeros((1,), dtype=jnp.float32),
                          "bar_grazing_energy": jnp.zeros((1,), dtype=jnp.float32),
                          "grazing_energy_sum": agent.state.content['grazing_energy_sum'], # VERY IMP: we need this for fitness
                          
                          "exchange_energy": jnp.zeros((1,), dtype=jnp.float32),
                          "bar_exchange_energy": jnp.zeros((1,), dtype=jnp.float32),

                          "bar_energy_taken": jnp.zeros((1,), dtype=jnp.float32), # energy taken from neighbors in the current step, used for normalization in the selector network
                          "abs_energy_exchange_sum": jnp.zeros((1,), dtype=jnp.float32), # absolute sum of energy exchanged with neighbors in the current step, used for normalization in the selector network

                          "metabolic_cost": jnp.zeros((1,), dtype=jnp.float32),
                          "bar_metabolic_cost": jnp.zeros((1,), dtype=jnp.float32),
                          "metabolic_cost_sum": agent.state.content['metabolic_cost_sum'], # VERY IMP: we need this for fitness
                          
                          "time_death": jnp.zeros((1,), dtype=jnp.float32), # time below death energy threshold
                          "time_reproduction": jnp.zeros((1,), dtype=jnp.float32), # time above reproduction energy threshold
                          "reproduce_flag": False, # flag to indicate if the agent is ready to reproduce
                          "death_flag": False, # flag to indicate if the agent is ready to die

                          "role": jnp.array([1.0]), # assign default role to child, can be changed later based on selector output, 0.0-> inactive, 1.0-> active/sunoptimal, 2.0->active/grazer, 3.0->active/exchanger
                          "num_switches": jnp.zeros((1,), dtype=jnp.float32)
                        }
        state = State(content = state_content)

        # update policy
        policy_params_content = { 'J': child_J,
                                 'tau': agent_to_copy.policy.params.content['tau'], # everyone has the same time constants
                                 'E': agent_to_copy.policy.params.content['E'], # everyone has the same mapping from observations to neurons
                                 'B': agent_to_copy.policy.params.content['B'], # everyone has the same bias for each neuron
                                 'D': agent_to_copy.policy.params.content['D']  # everyone has the same readout from neurons to actions
                                }
        policy_params = Params(content = policy_params_content)

        child_policy = CTRNN.set_policy(agent.policy, policy_params)
        child_policy = CTRNN.reset_policy(child_policy) # reset child's policy state

        return agent.replace(state = state, policy = child_policy, age = 0.0, active_state = 1, key = key)

    @staticmethod
    def remove_agent(agent, remove_params):
        state_content = {**agent.state.content,
                         'energy': jnp.zeros((1,), dtype=jnp.float32), # set energy to 0 on death, other state variables will be ignored when energy is 0
                         'x': jnp.array([MAX_WORLD_X + 2*RAY_LENGTH]), # move agent out of the world bounds
                         'y': jnp.array([MAX_WORLD_Y + 2*RAY_LENGTH]),
                         'x_dot': jnp.zeros((1,), dtype=jnp.float32),
                         'y_dot': jnp.zeros((1,), dtype=jnp.float32),
                         'ang_dot': jnp.zeros((1,), dtype=jnp.float32),
                         'l_acc': jnp.zeros((1,), dtype=jnp.float32),
                         'a_acc': jnp.zeros((1,), dtype=jnp.float32),
                         'movement': jnp.zeros((1,), dtype=jnp.float32),
                         'role': jnp.array([0.0]), # set to inactive
                         'time_reproduction': jnp.zeros((1,), dtype=jnp.float32),
                         'reproduce_flag': False,
                         'time_death': jnp.zeros((1,), dtype=jnp.float32), 
                         'death_flag': False} # set energy to 0 on death, other state variables will be ignored when energy is 0
        new_state = State(content = state_content)
        return agent.replace(state = new_state, age=0.0, active_state = 0)
    
    @staticmethod
    def half_parent_energy(agent, params):
        energy = agent.state.content['energy']
        new_energy = energy/2.0
        new_bar_energy = new_energy # bar energy is also halved
        state_content = {**agent.state.content, 
                         'time_reproduction': jnp.zeros((1,), dtype=jnp.float32), 
                         'reproduce_flag': False, 
                         'energy': new_energy, 
                         'bar_energy': new_bar_energy
                         }
        new_state = State(content = state_content)
        return agent.replace(state = new_state)







                          


