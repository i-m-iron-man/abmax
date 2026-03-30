import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'


from abmax.functions import *
import jax.numpy as jnp
import jax.random as random
import jax
from flax import struct
from evosax import CMA_ES

from sim_params import *
from agent import *
from neuron_policy import *
from ray_sensing import *
from utils import *

@struct.dataclass
class Boid_world():
    boid_set: Set
    selector_network: Selector_Network
    key: jax.random.PRNGKey

    @staticmethod
    def create_world(params, key):
        # get the data
        policy_params = params.content['policy_params']
        boid_params = params.content['agent_params']

        # create policies
        num_boids = boid_params['num_agents']
        num_active_boids = boid_params['num_active_agents']

        # create the boids
        key, *policy_keys = random.split(key, num_boids + 1)
        policy_keys = jnp.array(policy_keys)

        policy_create_params = Params(content={ 'num_neurons': policy_params['num_neurons'],
                                                'num_obs': policy_params['num_obs'],
                                                'num_actions': policy_params['num_actions']})
        
        policies = jax.vmap(CTRNN.create_policy, in_axes=(None, 0))(policy_create_params, policy_keys)  
        key, boid_key = jax.random.split(key)
        boid_create_params = Params(content={'policy': policies})

        boids = create_agents(agent=Boid, params=boid_create_params, num_agents=num_boids, num_active_agents=num_active_boids,
                               agent_type = boid_params['agent_type'], key=boid_key)
        
        boid_set = Set(num_agents=num_boids, num_active_agents=num_active_boids, agents=boids, id=0, set_type=boid_params['agent_type'], 
                           params=None, state=None, policy=None, key=None)
        
        # create selector network
        key, selector_key = jax.random.split(key)
        selector_network = Selector_Network.create_policy(selector_key)

        return Boid_world(boid_set = boid_set, selector_network = selector_network, key = key)
    
    @staticmethod
    def get_child_J(parent_agent, selector_network, key):
        
        parent_J = parent_agent.policy.params.content['J']
        
        key, *subkeys = random.split(key, parent_J.shape[0]*parent_J.shape[1]+1) # need one subkey for each connection in J plus one for noise
        subkeys = jnp.array(subkeys).reshape(parent_J.shape[0], parent_J.shape[1], 2) # reshape subkeys to match J shape, each connection gets 2 subkeys, one for selector input and one for noise
        
        parent_bar_Z = parent_agent.policy.state.content['bar_Z']
        parent_bar_energy = parent_agent.state.content['bar_energy']
        parent_bar_grazing_energy = parent_agent.state.content['bar_grazing_energy']
        parent_bar_exchange_energy = parent_agent.state.content['bar_exchange_energy']
        parent_bar_metabolic_cost = parent_agent.state.content['bar_metabolic_cost']
        parent_bar_movement = parent_agent.state.content['movement']


        def for_each_output_neuron(z_out, z_in_row, J_row, subkeys_row):
            def for_each_input_neuron(z_out, z_in, J_el, subkey):
                selector_input = jnp.concatenate([parent_bar_energy, parent_bar_grazing_energy, parent_bar_exchange_energy, 
                                                  parent_bar_metabolic_cost, parent_bar_movement, z_out.reshape(1), z_in.reshape(1), J_el.reshape(1)]).reshape(-1) # input to selector network for each connection
                selector_input = Signal(content={'input': selector_input})
                d_J_el = Selector_Network.step(selector_network, selector_input, None) # output of selector network for each connection
                J_noise = SELECTOR_NOISE_SCALE*jax.random.truncated_normal(subkey, lower=-1.0, upper=1.0, shape=()) # noise to add to each connection
                
                new_J_el = J_el + d_J_el * (1.0 + J_noise) # modulate the parent's J by the selector output to get child's J
                return jnp.clip(new_J_el, -J_LIMIT, J_LIMIT) # clip child's J to limits
            
            return jax.vmap(for_each_input_neuron, in_axes=(None, 0, 0, 0))(z_out, z_in_row, J_row, subkeys_row)
        child_J = jax.vmap(for_each_output_neuron, in_axes=(0, None, 0, 0))(parent_bar_Z, parent_bar_Z, parent_J, subkeys) # compute child's J by iterating over each connection in parent's J and applying the selector network

        return child_J
    
    @staticmethod
    def step_world(boid_world, _t):
        boid_set = boid_world.boid_set

        # get the agent interactions
        is_in_sums, energy_deltas = jit_agent_interactions(boid_set.agents) # get energy transfer between boids and whether they are in snatching distance, also get average movement for grazing energy calculation

        # get external sensor data
        ext_obs = jit_get_sensor_data(boid_set.agents)
        
        # step agents
        inputs_content = { 'ext_obs': ext_obs, 'energy_delta': energy_deltas, 'is_in_sum': is_in_sums }
        inputs = Signal(content=inputs_content)
        
        boid_step_params = Params(content={'dt': DT, 
                                            'action_scale': ACTION_SCALE,
                                            'time_constant_scale': NEURON_TIME_CONSTANT_SCALE,
                                            'avg_movement': get_avg_movement(boid_set.agents) # shape is (1,) 
                                            })
        
        boid_set = jit_step_agents(Boid.step_agent, boid_step_params, inputs, boid_set)

        # remove agents
        dead_boid_mask = jnp.where(boid_set.agents.state.content['death_flag'] == True, 1, 0) # get mask of agents that should be removed
        mask_params = Params(content={'set_mask': dead_boid_mask})
        boid_set = jit_set_agents_mask(Boid.remove_agent, 
                                       set_params=None, 
                                       mask_params=mask_params, 
                                       num_agents=-1, 
                                       set=boid_set)
        
        # add agents
        select_mask = jnp.where(boid_set.agents.active_state == False, 1, 0) # only select inactive agents for reproduction
        change_mask = boid_set.agents.state.content['reproduce_flag']
        mask_params = Params(content={'select_mask': select_mask, 'change_mask': change_mask})

        key, *child_keys = random.split(boid_world.key, NUM_AGENTS + 1) # need one key for each agent to reproduce plus one for the new world key
        child_keys = jnp.array(child_keys)
        child_Js = jax.vmap(Boid_world.get_child_J, in_axes=(0, None, 0))(boid_set.agents, boid_world.selector_network, child_keys) # get child Js for all parents that want to reproduce
        
        add_params = Params(content={'agent_to_copy': boid_set.agents, 'child_J': child_Js})
        boid_set, num_changes = jit_set_agents_rank_match(Boid.add_agent, 
                                                          set_params=add_params, 
                                                          mask_params=mask_params, 
                                                          num_agents=-1, 
                                                          set=boid_set)
        
        # half the energy of the parents that gave birth to the new agents
        mask_params = Params(content={'set_mask':change_mask})
        # we can use num_changes here as it is the number of parents that were selected
        boid_set = jit_set_agents_mask(Boid.half_parent_energy, 
                                       set_params=None, 
                                       mask_params=mask_params, 
                                       num_agents=num_changes, 
                                       set=boid_set)
        
        #rendering data
        '''
        render_data = Params(content={'xs': boid_set.agents.state.content['x'],
                                      'ys': boid_set.agents.state.content['y'],
                                      'angs': boid_set.agents.state.content['ang'],
                                      'energies': boid_set.agents.state.content['energy'],
                                      'grazing_energies': boid_set.agents.state.content['grazing_energy'],
                                      'exchange_energies': boid_set.agents.state.content['exchange_energy'],
                                      'metabolic_costs': boid_set.agents.state.content['metabolic_cost'],
                                      'roles': boid_set.agents.state.content['role']
        })
        '''
        total_age_mask = jnp.multiply(boid_set.agents.active_state, 1 - boid_set.agents.params.content['immortal_flag']) # only consider age for active and mortal agents
        agent_ages = jnp.multiply(boid_set.agents.age, total_age_mask) # get age of agents that should be considered for fitness calculation
        total_age = jnp.sum(agent_ages) # sum age across all agents to get total age for fitness calculation
        
        return boid_world.replace(boid_set = boid_set, key=key), (total_age, boid_set.num_active_agents-NUM_ACTIVE_AGENTS, None)#, render_data) # return the updated world and the fitness components for logging purposes (total age and number of active agents above the initial number)
    
    @staticmethod
    def reset_world(boid_world):
        boids = boid_world.boid_set.agents
        boids = jax.vmap(Boid.reset_agent)(boids, None) # reset all agents to initial state
        boid_set = boid_world.boid_set.replace(agents=boids, num_active_agents=NUM_ACTIVE_AGENTS)
        return boid_world.replace(boid_set=boid_set)

        
jit_step_world = jax.jit(Boid_world.step_world)
jit_reset_world = jax.jit(Boid_world.reset_world)
        
def scan_episode(boid_world:Boid_world, ts):
    """
    Scan the boid world for a given number of time steps.
    Args:
        - boid_world: The boid world to scan. this is the carry value
        - ts: The time steps to scan
    Returns:
        - The updated boid world after the time steps
        - the render_data for the boids and the patches: (forsger_xs: (num_boids, ts), boid_ys: (num_boids, ts), boid_angs: (num_boids, ts), patch_energies: (num_patches, ts))
    """
    return jax.lax.scan(jit_step_world, boid_world, ts) # scan(Scanning_function, carry, xs) where xs is the time steps to scan
    #return jax.lax.scan(Boid_world.step_world, boid_world, ts)
jit_scan_episode = jax.jit(scan_episode)

def run_episode(boid_world:Boid_world):
    """
    run the episode. This function is used as a wrapper for the scan_episode function so that a jitted version can be created
    Args:
        - boid_world: The boid world to run the episode
    Returns:
        - The updated boid world after the episode
        - the render_data for the boids and the patches: (boid_xs: (num_boids, ts), boid_ys: (num_boids, ts), boid_angs: (num_boids, ts), patch_energies: (num_patches, ts))
    """
    ts = jnp.arange(EP_LEN)
    boid_world = jit_reset_world(boid_world)  # reset the world before running the episode
    boid_world, (total_age, num_active_agents, render_data) = jit_scan_episode(boid_world, ts)    
    
    return boid_world, total_age, num_active_agents#, render_data

jit_run_episode = jax.jit(run_episode)

def set_CMAES_params(CMAES_params, boid_world):
    # set one vector.member of cmaes param over one single world
    E = CMAES_params[:NUM_NEURONS * NUM_OBS].reshape(NUM_NEURONS, NUM_OBS)
    last_index = NUM_NEURONS * NUM_OBS

    D = CMAES_params[last_index:last_index + NUM_NEURONS*NUM_ACTIONS].reshape(NUM_ACTIONS, NUM_NEURONS)
    last_index += NUM_NEURONS*NUM_ACTIONS

    B = CMAES_params[last_index:last_index + NUM_NEURONS].reshape(NUM_NEURONS,)
    last_index += NUM_NEURONS

    tau = CMAES_params[last_index:last_index + NUM_NEURONS].reshape(NUM_NEURONS,)
    last_index += NUM_NEURONS

    w1 = CMAES_params[last_index:last_index + MLP_SELECTOR_HIDDEN_SIZE*NUM_INPUTS_SELECTOR].reshape(MLP_SELECTOR_HIDDEN_SIZE, NUM_INPUTS_SELECTOR)
    last_index += MLP_SELECTOR_HIDDEN_SIZE*NUM_INPUTS_SELECTOR

    b1 = CMAES_params[last_index:last_index + MLP_SELECTOR_HIDDEN_SIZE].reshape(MLP_SELECTOR_HIDDEN_SIZE,)
    last_index += MLP_SELECTOR_HIDDEN_SIZE

    w2 = CMAES_params[last_index:last_index + MLP_SELECTOR_HIDDEN_SIZE*MLP_SELECTOR_HIDDEN_SIZE].reshape(MLP_SELECTOR_HIDDEN_SIZE, MLP_SELECTOR_HIDDEN_SIZE)
    last_index += MLP_SELECTOR_HIDDEN_SIZE*MLP_SELECTOR_HIDDEN_SIZE

    b2 = CMAES_params[last_index:last_index + MLP_SELECTOR_HIDDEN_SIZE].reshape(MLP_SELECTOR_HIDDEN_SIZE,)
    last_index += MLP_SELECTOR_HIDDEN_SIZE

    w3 = CMAES_params[last_index:last_index + MLP_SELECTOR_HIDDEN_SIZE].reshape(MLP_SELECTOR_HIDDEN_SIZE,)
    last_index += MLP_SELECTOR_HIDDEN_SIZE
    
    key, subkey = jax.random.split(boid_world.key)
    J = jax.random.uniform(subkey, shape=(NUM_NEURONS, NUM_NEURONS), minval=-1.0, maxval=1.0, dtype=jnp.float32) # initialize J randomly as it is not evolved directly but through the selector network
    
    CTRNN_policy_params = Params(content={'E': E, 'D': D, 'B': B, 'tau': tau, 'J': J})
    selector_network_params = Params(content={'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3})

    new_selector_network = Selector_Network.set_policy(boid_world.selector_network, selector_network_params)
    
    new_boid_policies = jax.vmap(CTRNN.set_policy, in_axes=(0, None))(boid_world.boid_set.agents.policy, CTRNN_policy_params) # vmaps over all agents
    new_boids = boid_world.boid_set.agents.replace(policy=new_boid_policies)
    new_boid_set = boid_world.boid_set.replace(agents=new_boids)

    return boid_world.replace(selector_network=new_selector_network, boid_set=new_boid_set, key=key)

jit_set_CMAES_params = jax.jit(set_CMAES_params)



def get_fitness(CMAES_params, boid_worlds:Boid_world): 
    # CMAES params: (NUM_WORLDS, NUM_ES_PARAMS)
    # boid_worlds: (NUM_WORLDS, NUM_SCENARIOS, ...)
    # returns fitness: (NUM_WORLDS,)

    boid_worlds = jax.vmap(jax.vmap(set_CMAES_params, in_axes=(None, 0)), in_axes=(0, 0))(CMAES_params, boid_worlds) # set CMAES params for each world
    boid_worlds, total_age_series, num_active_agents_series= jax.vmap(jax.vmap(jit_run_episode))(boid_worlds) # run episode for each world and scenario
    
    # age related data
    total_age_series_sum = jnp.sum(total_age_series, axis=2) # sum total age across scenarios for each world, shape: (NUM_WORLDS, NUM_SCENARIOS)
    age_fitness = jnp.mean(total_age_series_sum/(EP_LEN*NUM_AGENTS*NUM_AGENTS), axis=1) # average the average age per agent across scenarios for each world to get the fitness, shape: (NUM_WORLDS,) 

    # Num_agent related data
    max_num_active_agents = jnp.max(num_active_agents_series, axis=2) # max number of active agents across scenarios for each world, shape: (NUM_WORLDS, NUM_SCENARIOS)
    max_num_active_agents = jnp.mean(max_num_active_agents, axis=1) # average max number of active agents across scenarios for each world, shape: (NUM_WORLDS,)
    max_num_active_agents = jnp.mean(max_num_active_agents) # average max number of active agents across worlds, shape: (1,) this is just for logging purposes

    # total energy gained related data
    total_grazing_energy_sum = jnp.sum(boid_worlds.boid_set.agents.state.content['grazing_energy_sum'], axis=2) # shape: (NUM_WORLDS, NUM_SCENARIOS, 1)
    total_metabolic_cost_sum = jnp.sum(boid_worlds.boid_set.agents.state.content['metabolic_cost_sum'], axis=2) # shape: (NUM_WORLDS, NUM_SCENARIOS, 1)
    
    total_e_gained = total_grazing_energy_sum - total_metabolic_cost_sum # compute fitness for each world and scenario, shape: (NUM_WORLDS, NUM_SCENARIOS, 1)
    energy_fitness = jnp.mean(total_e_gained/EP_LEN, axis=1).reshape(-1,) # average fitness across scenarios for each world, shape: (NUM_WORLDS, 1)

    
    # final fitness calculation
    fitness = age_fitness + 0.5*energy_fitness # combine age fitness and num agent fitness to get final fitness, shape: (NUM_WORLDS,)
    
    #logging data
    avg_energy_fitness = jnp.mean(energy_fitness) # average energy fitness across worlds, shape: (1,) this is just for logging purposes
    
    avg_interaction_energy = jnp.mean(boid_worlds.boid_set.agents.state.content['abs_energy_exchange_sum'], axis=2).reshape(-1,) # average interaction energy across scenarios for each world, shape: (NUM_WORLDS,)
    avg_interaction_energy = jnp.mean(avg_interaction_energy/EP_LEN) # average interaction energy across worlds, shape: (1,) this is just for logging purposes
    
    avg_age_fitness = jnp.mean(age_fitness) # average age fitness across worlds, shape: (1,) this is just for logging purposes

    return fitness, boid_worlds, avg_energy_fitness, avg_age_fitness, avg_interaction_energy, max_num_active_agents

jit_get_fitness = jax.jit(get_fitness)



def main():
    key, *boid_world_keys = random.split(KEY, NUM_WORLDS * NUM_SCENARIOS + 1) # NUM_WORLDS for parallel worlds, NUM_SCENARIOS for different initial conditions
    boid_world_keys = jnp.array(boid_world_keys).reshape(NUM_WORLDS, NUM_SCENARIOS, 2) # reshape keys to match worlds and scenarios
    sim_params = Params(content=SIM_PARAMS_CONTENT)

    #initialize boid worlds: (NUM_WORLDS, NUM_SCENARIOS, ...)
    boid_worlds = jax.vmap(jax.vmap(Boid_world.create_world, in_axes=(None, 0)), in_axes=(None, 0))(sim_params, boid_world_keys) # create boid worlds for each scenario and world, shape: (NUM_WORLDS, NUM_SCENARIOS, ...)
    
    #initialize CMAES
    key, cmaes_key = random.split(key, 2)
    strategy = CMA_ES(popsize=POPULATION_SIZE, num_dims=NUM_ES_PARAMS, elite_ratio=ELITE_RATIO, sigma_init= SIGMA_INIT)
    es_params = strategy.default_params
    state = strategy.initialize(cmaes_key, es_params)

    #initialize lists to save fitness and params
    mean_fitness_list = [] # progression of the mean error over generations
    avg_age_fitness_list = [] # progression of the average age fitness over generations
    avg_energy_fitness_list = [] # progression of the average energy fitness over generations
    saved_fitness_list = [] # the errors at which params were saved
    saved_generation_list = [] # the generation numbers at which params were saved
    param_list_mean = [] # the saved params
    params_list_best = [] # the saved params
    total_e_gained_list = [] # progression of the total energy gained over generations
    avg_interaction_energy_list = [] # progression of the average interaction energy over generations

    # training loop
    for generation in range(NUM_GENERATIONS):
        key, gen_key = jax.random.split(key, 2)
        x, state = strategy.ask(gen_key, state, es_params)
        fitness, boid_worlds, avg_energy_fitness, avg_age_fitness,avg_interaction_energy, max_num_active_agents = jit_get_fitness(x, boid_worlds)
        state = strategy.tell(x, -1*fitness, state, es_params)

        mean_fitness = jnp.mean(fitness)

        mean_fitness_list.append(mean_fitness)
        avg_energy_fitness_list.append(avg_energy_fitness)
        avg_age_fitness_list.append(avg_age_fitness)
        total_e_gained_list.append(avg_energy_fitness)
        avg_interaction_energy_list.append(avg_interaction_energy)

        print('Generation:', generation, 'Mean Fitness:', mean_fitness, 'Avg Age Fitness:', avg_age_fitness, 
              'Avg Energy Fitness:', avg_energy_fitness, 'Avg Interaction Energy:', avg_interaction_energy, 
              'Max Active Agents:', max_num_active_agents)
        
        if generation == NUM_GENERATIONS - 1 or generation == 0 or generation == int(NUM_GENERATIONS/2) or generation == int(NUM_GENERATIONS/4) or generation == int(3*NUM_GENERATIONS/4):
            saved_generation_list.append(generation)
            param_list_mean.append(state.mean)
            best_idx = jnp.argmax(fitness)
            params_list_best.append(x[best_idx])
    
    param_list_mean = jnp.array(param_list_mean)
    params_list_best = jnp.array(params_list_best)
    saved_fitness_list = jnp.array(saved_fitness_list)
    saved_generation_list = jnp.array(saved_generation_list)
    total_e_gained_list = jnp.array(total_e_gained_list)
    avg_interaction_energy_list = jnp.array(avg_interaction_energy_list)
    avg_age_fitness_list = jnp.array(avg_age_fitness_list)
    avg_energy_fitness_list = jnp.array(avg_energy_fitness_list)

    mean_fitness_list = jnp.array(mean_fitness_list)

    jnp.save(PARAM_PATH + 'params_list_mean.npy', param_list_mean)
    jnp.save(PARAM_PATH + 'params_list_best.npy', params_list_best)
    jnp.save(PARAM_PATH + 'saved_fitness_list.npy', saved_fitness_list)
    jnp.save(PARAM_PATH + 'saved_generation_list.npy', saved_generation_list)
    jnp.save(PARAM_PATH + 'mean_fitness_progress.npy', mean_fitness_list)
    jnp.save(PARAM_PATH + 'total_e_gained_list.npy', total_e_gained_list)
    jnp.save(PARAM_PATH + 'avg_interaction_energy_list.npy', avg_interaction_energy_list)
    jnp.save(PARAM_PATH + 'avg_age_fitness_list.npy', avg_age_fitness_list)
    jnp.save(PARAM_PATH + 'avg_energy_fitness_list.npy', avg_energy_fitness_list)
    jnp.save(PARAM_PATH + 'test_key.npy', jnp.array(key))

if __name__ == "__main__":
    main()



