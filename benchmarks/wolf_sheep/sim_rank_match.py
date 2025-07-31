import sys
sys.path.append('/Users/siddarth.chaturvedi/Desktop/source/abmax_git/abmax')
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
palette = "viridis"
sns.set_palette(palette)


from structs import *
from functions import *
from flax import struct
'''small world params
'''
KEY = jax.random.PRNGKey(0)
SIM_STEPS = 100
X_POS_MAX = 100
Y_POS_MAX = 100
NUM_ECOSYSTEMS = 10

#Wolf Params
NUM_WOLVES = 1200
NUM_ACTIVE_WOLVES = 400
REPRO_PROBAB_WOLVES = 0.09
DELTA_ENERGY_WOLVES = 30.0
WOLF_AGENT_TYPE = 2

#Sheep Params
NUM_SHEEP = 1200
NUM_ACTIVE_SHEEP = 600
REPRO_PROBAB_SHEEP = 0.01
DELTA_ENERGY_SHEEP = 5.0
SHEEP_AGENT_TYPE = 1

#grass Params
REGROWTH_TIME = 20
NUM_GRASS = X_POS_MAX * Y_POS_MAX
GRASS_AGENT_TYPE = 0


'''Large world params'''
'''
KEY = jax.random.PRNGKey(0)
SIM_STEPS = 100
X_POS_MAX = 1000
Y_POS_MAX = 1000
NUM_ECOSYSTEMS = 10

#Wolf Params
NUM_WOLVES = 12000
NUM_ACTIVE_WOLVES = 4000
REPRO_PROBAB_WOLVES = 0.09
DELTA_ENERGY_WOLVES = 30.0
WOLF_AGENT_TYPE = 2

#Sheep Params
NUM_SHEEP = 12000
NUM_ACTIVE_SHEEP = 6000
REPRO_PROBAB_SHEEP = 0.01
DELTA_ENERGY_SHEEP = 5.0
SHEEP_AGENT_TYPE = 1

#grass Params
REGROWTH_TIME = 20
NUM_GRASS = X_POS_MAX * Y_POS_MAX
GRASS_AGENT_TYPE = 0
'''
ECOSYSTEM_PARAMS = Params(content={'grass_params': {'regrowth_time': REGROWTH_TIME, 'x_max': X_POS_MAX, 'y_max': Y_POS_MAX, 
                                                    'agent_type': GRASS_AGENT_TYPE, 'num_grass': NUM_GRASS},
                                    'wolf_params': {'num_wolves': NUM_WOLVES, 'num_active_wolves': NUM_ACTIVE_WOLVES,
                                                    'reproduction_probab': REPRO_PROBAB_WOLVES, 
                                                    'delta_energy': DELTA_ENERGY_WOLVES, 
                                                    'agent_type': WOLF_AGENT_TYPE},
                                    'sheep_params': {'num_sheeps': NUM_SHEEP, 'num_active_sheeps': NUM_ACTIVE_SHEEP,
                                                    'reproduction_probab': REPRO_PROBAB_SHEEP,
                                                    'delta_energy': DELTA_ENERGY_SHEEP,
                                                    'agent_type': SHEEP_AGENT_TYPE}})


@struct.dataclass
class Animal(Agent):

    @staticmethod
    def create_agent(type, params, id, active_state, key):
        key, subkey = jax.random.split(key) 
        
        X_pos_max = params.content['x_max']
        Y_pos_max = params.content['y_max']

        agent_params_content = {'reproduction_probab': params.content['reproduction_probab'], 
                                'delta_energy': params.content['delta_energy'], 'X_pos_max': X_pos_max, 
                                'Y_pos_max': Y_pos_max}
        
        agent_params = Params(content=agent_params_content)

        def create_active_agent():
            key, *create_keys = jax.random.split(subkey, 4)

            X_pos = jax.random.randint(create_keys[0], minval=0, maxval=X_pos_max-1, shape=(1,))
            Y_pos = jax.random.randint(create_keys[1], minval=0, maxval=Y_pos_max-1, shape=(1,))
            
            energy = jax.random.uniform(create_keys[2], minval=1.0, maxval=2.0*params.content['delta_energy'], shape=(1,))

            state_content = {'X_pos': X_pos, 'Y_pos': Y_pos, 'energy': energy, 'reproduce': 0}
            state = State(content=state_content)
            return state
        
        def create_inactive_agent():
            state_content = {'X_pos': jnp.array([-1]), 'Y_pos': jnp.array([-1]), 'energy': jnp.array([-1.0]), 'reproduce': 0}
            state = State(content=state_content)
            return state
        
        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(), 
                                   lambda _: create_inactive_agent(), None)
        return Animal(id=id, active_state=active_state, age=0.0, agent_type=type,
                      params=agent_params, state=agent_state, policy=None, key=key)
    
    @staticmethod
    def step_agent(agent, input, step_params):
        def step_active_agent():
            energy_in = input.content['energy_in']
            X_pos = agent.state.content['X_pos']
            Y_pos = agent.state.content['Y_pos']
            energy = agent.state.content['energy']

            key, subkey = jax.random.split(agent.key)
            action = jax.random.randint(subkey, minval=0, maxval=4, shape=(1,))

            # Move the animal
            X_pos_new = jax.lax.cond(action[0] == 0, lambda _: X_pos + 1, 
                                    lambda _: X_pos, None)
            X_pos_new = jax.lax.cond(action[0] == 1, lambda _: X_pos - 1,
                                    lambda _: X_pos_new, None)
            Y_pos_new = jax.lax.cond(action[0] == 2, lambda _: Y_pos + 1,
                                    lambda _: Y_pos, None)
            Y_pos_new = jax.lax.cond(action[0] == 3, lambda _: Y_pos - 1,
                                    lambda _: Y_pos_new, None)
            
            # check for boundaries
            X_pos_new = jnp.clip(X_pos_new, 0, agent.params.content['X_pos_max']-1)
            Y_pos_new = jnp.clip(Y_pos_new, 0, agent.params.content['Y_pos_max']-1)

            energy_new = energy - 1.0 + agent.params.content['delta_energy']*energy_in

            key, reproduce_key = jax.random.split(key)
            rand_float = jax.random.uniform(reproduce_key, shape=(1,))
            reproduce = jax.lax.cond(rand_float[0] < agent.params.content['reproduction_probab'], lambda _: 1, 
                                    lambda _: agent.state.content['reproduce'], None)
            
            new_state_content = {'X_pos': X_pos_new, 'Y_pos': Y_pos_new, 'energy': energy_new, 'reproduce': reproduce}
            new_state = State(content = new_state_content)
            return agent.replace(state = new_state, age = agent.age + 1.0, key=key)
        
        def step_inactive_agent():
            return agent
        
        new_agent = jax.lax.cond(agent.active_state, lambda _: step_active_agent(), lambda _: step_inactive_agent(), None)
        return new_agent
    
    @staticmethod
    def remove_agent(agent, remove_params):
        state_content = {'X_pos': jnp.array([-1]), 'Y_pos': jnp.array([-1]),
                         'energy': jnp.array([-1.0]), 'reproduce': 0}
        state = State(content=state_content)
        return agent.replace(state=state, active_state=0, age=0.0)
    
    @staticmethod
    def add_agent(agent, add_params):
        agent_to_copy = add_params.content['agent_to_copy']
        
        X_pos = agent_to_copy.state.content['X_pos']
        Y_pos = agent_to_copy.state.content['Y_pos']
        energy = agent_to_copy.state.content['energy']/2
        
        state_content = {'X_pos': X_pos, 'Y_pos': Y_pos, 'energy': energy, 'reproduce': 0}
        state = State(content=state_content)
        age = 0.0
        active_state = 1
        return agent.replace(state=state, age=age, active_state=active_state)
    
    @staticmethod
    def half_energy(agent, set_params):
        state_content = {'X_pos': agent.state.content['X_pos'], 
                         'Y_pos': agent.state.content['Y_pos'], 
                         'energy': agent.state.content['energy']/2, 
                         'reproduce': agent.state.content['reproduce']}
        state = State(content=state_content)
        return agent.replace(state=state)
    


@struct.dataclass
class Grass(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
        regrowth_time = params.content['regrowth_time']
        
        X_pos_max = params.content['x_max']
        params_content = {'regrowth_time': regrowth_time}
        params = Params(content=params_content)

        
        
        X_pos = jnp.array([jnp.mod(id, X_pos_max)])
        Y_pos = jnp.array([jnp.floor_divide(id, X_pos_max)])

        key, fully_grown_key, count_down_key = jax.random.split(key, 3)
        fully_grown = jax.random.choice(fully_grown_key, a=jnp.array([True, False]), shape=(1,))
        count_down = jax.lax.cond(fully_grown[0], 
                                  lambda _: jnp.array([0]), 
                                  lambda _: jax.random.randint(count_down_key, minval=1, maxval=regrowth_time, shape=(1,)), 
                                  None)

        state_content = {'X_pos': X_pos, 'Y_pos': Y_pos, 'fully_grown': fully_grown, 'count_down': count_down}
        state = State(content=state_content)
        age = 0.0
        return Grass(id=id, active_state=active_state, age=age, agent_type=type, 
                     params=params, state=state, policy=None, key=key)
    
    @staticmethod
    def step_agent(agent, input, step_params):
        energy_out = input.content['energy_out'] # the logic of fully grown grass gets eaten is in the main step function
        count_down = agent.state.content['count_down']
        fully_grown = agent.state.content['fully_grown']
        
        new_count_down, new_fully_grown = jax.lax.cond(energy_out[0], lambda _: (jnp.array([agent.params.content['regrowth_time']]), jnp.array([False])), 
                                                       lambda _: (count_down, fully_grown), None)
        
        new_count_down = jax.lax.cond(new_fully_grown[0], lambda _: new_count_down, lambda _: new_count_down - 1, None)
        
        new_fully_grown = jax.lax.cond(new_count_down[0] <= 0, lambda _: jnp.array([True]), lambda _: new_fully_grown, None)

        new_grass_state_content = {'X_pos': agent.state.content['X_pos'], 
                                   'Y_pos': agent.state.content['Y_pos'], 
                                   'fully_grown': new_fully_grown, 
                                   'count_down': new_count_down}
        
        new_grass_state = State(content=new_grass_state_content)

        new_grass = agent.replace(state=new_grass_state)
        return new_grass


def interaction(wolves:Animal, sheeps:Animal, grasses:Grass):
    
    # wolves eat sheep:
    def one_wolf_all_sheep(wolf, sheeps): # vmap across all wolves

        def one_wolf_one_sheep(wolf, sheep): # vmap across all sheep
            wolf_X_pos = wolf.state.content['X_pos']
            wolf_Y_pos = wolf.state.content['Y_pos']
            sheep_X_pos = sheep.state.content['X_pos']
            sheep_Y_pos = sheep.state.content['Y_pos']
            condition = jnp.logical_and(wolf_X_pos[0] == sheep_X_pos[0], wolf_Y_pos[0] == sheep_Y_pos[0])
            
            wolf_energy_in = jax.lax.cond(condition, lambda _: jnp.array([1.0]), lambda _: jnp.array([0.0]), None)
            return wolf_energy_in
        
        one_wolf_energy_from_all_sheeps = jax.vmap(one_wolf_one_sheep, in_axes=(None, 0))(wolf, sheeps)
        return one_wolf_energy_from_all_sheeps
    
    wolves_sheeps_matrix = jax.vmap(one_wolf_all_sheep, in_axes=(0, None))(wolves, sheeps)
    '''
    for wolves this matrix is summed across all columns to get the total energy gained by each wolf
    for sheeps we take the max of the matrix across all rows to get if the sheep is eaten or not
    '''
    wolves_energy_in = jnp.sum(wolves_sheeps_matrix, axis=1, dtype=jnp.int32)
    sheeps_eaten = jnp.max(wolves_sheeps_matrix, axis=0)

    # sheeps eat grass:
    def one_sheep_all_grass(sheep, grasses): # vmap across all sheeps

        def one_sheep_one_grass(sheep, grass): # vmap across all grass
            sheep_X_pos = sheep.state.content['X_pos']
            sheep_Y_pos = sheep.state.content['Y_pos']
            grass_X_pos = grass.state.content['X_pos']
            grass_Y_pos = grass.state.content['Y_pos']
            grass_fully_grown = grass.state.content['fully_grown'][0] # fully grown = jnp.array([True]) or jnp.array([False])
            condition = jnp.logical_and(sheep_X_pos[0] == grass_X_pos[0], sheep_Y_pos[0] == grass_Y_pos[0])
            condition = jnp.logical_and(condition, grass_fully_grown)

            sheep_energy_in = jax.lax.cond(condition, lambda _: jnp.array([1.0]), lambda _: jnp.array([0.0]), None)
            return sheep_energy_in
        one_sheep_energy_from_all_grass = jax.vmap(one_sheep_one_grass, in_axes=(None, 0))(sheep, grasses)
        return one_sheep_energy_from_all_grass
    
    sheeps_grass_matrix = jax.vmap(one_sheep_all_grass, in_axes=(0, None))(sheeps, grasses)
    '''
    for sheeps this matrix is summed across all columns to get the total energy gained by each sheep
    for grass we take the max of the matrix across all rows to get if the grass is eaten or not
    '''
    sheeps_energy_in = jnp.sum(sheeps_grass_matrix, axis=1, dtype=jnp.int32)
    grasses_eaten = jnp.max(sheeps_grass_matrix, axis=0)

    return wolves_energy_in, sheeps_energy_in, sheeps_eaten, grasses_eaten
jit_interaction = jax.jit(interaction)

    

@struct.dataclass
class Ecosystem():
    wolf_set: Set
    sheep_set: Set
    grass_set: Set

    @staticmethod
    def create_ecosystem(parmas, key):
        key, grass_key, wolf_key, sheep_key = jax.random.split(key, 4)

        grass_params, wolf_params, sheep_params = parmas.content['grass_params'], parmas.content['wolf_params'], parmas.content['sheep_params']

        num_grass = grass_params['num_grass']
        grass_agent_type = grass_params['agent_type']
        
        regrowth_time_arr = jnp.tile(grass_params['regrowth_time'], num_grass)
        x_max = grass_params['x_max']
        y_max = grass_params['y_max']
        x_max_arr = jnp.tile(grass_params['x_max'], num_grass)
        grass_create_params = Params(content={'regrowth_time': regrowth_time_arr, 'x_max': x_max_arr})

        grass_agents = create_agents(Grass, params=grass_create_params, num_agents=num_grass, 
                                     num_active_agents=num_grass, agent_type=grass_agent_type, key=grass_key)
        grass_set = Set(agents=grass_agents, num_agents=num_grass, state=None, params=None, policy=None, id=0,
                        num_active_agents=num_grass, key=0, set_type=grass_agent_type)
        
        num_wolves = wolf_params['num_wolves']
        num_active_wolves = wolf_params['num_active_wolves']
        wolf_agent_type = wolf_params['agent_type']

        reproduction_probab_arr = jnp.tile(wolf_params['reproduction_probab'], num_wolves)
        x_max_arr_wolves = jnp.tile(x_max, num_wolves)
        y_max_arr_wolves = jnp.tile(y_max, num_wolves)
        delta_energy_arr = jnp.tile(wolf_params['delta_energy'], num_wolves)
        wolf_create_params = Params(content={'reproduction_probab': reproduction_probab_arr, 
                                             'delta_energy': delta_energy_arr, 
                                             'x_max': x_max_arr_wolves, 
                                             'y_max': y_max_arr_wolves})
        
        wolf_agents = create_agents(Animal, params=wolf_create_params, num_agents=num_wolves,
                                     num_active_agents=num_active_wolves, agent_type=wolf_agent_type, key=wolf_key)
        wolf_set = Set(agents=wolf_agents, num_agents=num_wolves, state=None, params=None, policy=None, id=0,
                        num_active_agents=num_active_wolves, key=None, set_type=wolf_agent_type)
        
        num_sheeps = sheep_params['num_sheeps']
        num_active_sheeps = sheep_params['num_active_sheeps']
        sheep_agent_type = sheep_params['agent_type']

        reproduction_probab_arr_sheep = jnp.tile(sheep_params['reproduction_probab'], num_sheeps)
        x_max_arr_sheep = jnp.tile(x_max, num_sheeps)
        y_max_arr_sheep = jnp.tile(y_max, num_sheeps)
        delta_energy_arr_sheep = jnp.tile(sheep_params['delta_energy'], num_sheeps)
        sheep_create_params = Params(content={'reproduction_probab': reproduction_probab_arr_sheep, 
                                              'delta_energy': delta_energy_arr_sheep, 
                                              'x_max': x_max_arr_sheep, 
                                              'y_max': y_max_arr_sheep})
        sheep_agents = create_agents(Animal, params=sheep_create_params, num_agents=num_sheeps,
                                     num_active_agents=num_active_sheeps, agent_type=sheep_agent_type, key=sheep_key)
        sheep_set = Set(agents=sheep_agents, num_agents=num_sheeps, state=None, params=None, policy=None, id=0,
                        num_active_agents=num_active_sheeps, key=None, set_type=sheep_agent_type)

        return Ecosystem(wolf_set=wolf_set, sheep_set=sheep_set, grass_set=grass_set)
    
    @staticmethod
    def add_animals(animal_set:Set):
        # select child_slots
        select_mask = jnp.where(animal_set.agents.active_state == 0, 1, 0)
        #select parents
        change_mask = animal_set.agents.state.content['reproduce']
        
        mask_params = Params(content={'select_mask': select_mask, 'change_mask': change_mask})
        add_params = Params(content={'agent_to_copy': animal_set.agents})

        new_animal_set, num_changes = jit_set_agents_rank_match(Animal.add_agent , set_params=add_params, mask_params=mask_params, num_agents=-1, set=animal_set)

        # half the energy of the parents
        mask_params = Params(content={'set_mask':change_mask})
        # we can use num_changes here as it is the number of parents that were selected
        new_animal_set = jit_set_agents_mask(Animal.half_energy, set_params=None, mask_params=mask_params, num_agents=num_changes, set=new_animal_set)
        return new_animal_set
        

    @staticmethod
    def step(ecosystem, _t):
        wolves_energy_in, sheeps_energy_in, sheeps_eaten, grasses_eaten = jit_interaction(ecosystem.wolf_set.agents,
                                                                                           ecosystem.sheep_set.agents,
                                                                                           ecosystem.grass_set.agents)
        grass_step_signal = Signal(content={'energy_out': grasses_eaten})
        new_grass_set = jit_step_agents(Grass.step_agent, None, grass_step_signal, ecosystem.grass_set)

        sheep_step_signal = Signal(content={'energy_in': sheeps_energy_in})
        new_sheep_set = jit_step_agents(Animal.step_agent, None, sheep_step_signal, ecosystem.sheep_set)

        wolf_step_signal = Signal(content={'energy_in': wolves_energy_in})
        new_wolf_set = jit_step_agents(Animal.step_agent, None, wolf_step_signal, ecosystem.wolf_set)

        #remove the dead sheeps and wolves
        eaten_sheep_mask = sheeps_eaten.reshape(-1)
        hungry_sheep_mask = jnp.where(new_sheep_set.agents.state.content['energy'] <= 0.0, 1, 0).reshape(-1)
        dead_sheep_mask = jnp.logical_or(eaten_sheep_mask, hungry_sheep_mask)
        remove_params_sheep = Params(content={'set_mask': dead_sheep_mask})
        new_sheep_set = jit_set_agents_mask(Animal.remove_agent, set_params=None, mask_params=remove_params_sheep, num_agents=-1, set=new_sheep_set)

        dead_wolf_mask = jnp.where(new_wolf_set.agents.state.content['energy'] <= 0.0, 1, 0).reshape(-1)
        remove_params_wolf = Params(content={'set_mask': dead_wolf_mask})
        new_wolf_set = jit_set_agents_mask(Animal.remove_agent, set_params=None, mask_params=remove_params_wolf, num_agents=-1, set=new_wolf_set)

        #add new sheeps and wolves
        new_sheep_set = Ecosystem.add_animals(new_sheep_set)
        new_wolf_set = Ecosystem.add_animals(new_wolf_set)

        return Ecosystem(wolf_set=new_wolf_set, sheep_set=new_sheep_set, grass_set=new_grass_set), (new_wolf_set.num_active_agents, new_sheep_set.num_active_agents)
    
    
def run_scan(ecosystem,ts):
    ecosystem, (num_wolves, num_sheeps) = jax.lax.scan(Ecosystem.step, ecosystem, ts)
    return ecosystem, (num_wolves, num_sheeps)
jit_run_scan = jax.jit(run_scan)
jit_vmap_run_scan = jax.jit(jax.vmap(run_scan, in_axes=(0, None)))

def run_main(params, key):
    ecosystem = Ecosystem.create_ecosystem(params, key)
    ts = jnp.arange(SIM_STEPS)
    ecosystem, (num_wolves, num_sheeps) = jit_run_scan(ecosystem, ts)
    return ecosystem, (num_wolves, num_sheeps)

def run_main_vmap(params, key, num_ecosystems):
    key, *subkeys = jax.random.split(key, num_ecosystems + 1)
    subkeys = jnp.array(subkeys)
    ecosystems = jax.vmap(Ecosystem.create_ecosystem, in_axes=(None, 0))(params, subkeys)
    ts = jnp.arange(SIM_STEPS)
    ecosystems, (num_wolves, num_sheeps) = jit_vmap_run_scan(ecosystems, ts)
    return ecosystems, (num_wolves, num_sheeps)

def main():
    eco, (num_wolves, num_sheeps) = run_main(ECOSYSTEM_PARAMS, KEY)
    plt.plot(num_wolves, label='Wolves')
    plt.plot(num_sheeps, label='Sheep')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Agents')
    plt.title('Population Dynamics of Wolves and Sheep')
    plt.legend()
    plt.show()



def main_vmap():
    eco, (num_wolves, num_sheeps) = run_main_vmap(ECOSYSTEM_PARAMS, KEY, NUM_ECOSYSTEMS)
    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(NUM_ECOSYSTEMS):
        ax.plot(num_wolves[i], label=f'wolves ecosystem {i+1}')
    ax.set_xlabel('time steps', fontsize=50)
    ax.set_ylabel('number of wolves', fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.legend(fontsize=25)
    plt.savefig('./wolf_dynamics.svg', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 10))
    for i in range(NUM_ECOSYSTEMS):
        ax.plot(num_sheeps[i], label=f'sheep ecosystem {i+1}')
    ax.set_xlabel('time steps', fontsize=50)
    ax.set_ylabel('number of sheep', fontsize=50)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.legend(fontsize=25)
    plt.savefig('./sheep_dynamics.svg', bbox_inches='tight')
    plt.show()


    
    '''
    plt.figure(figsize=(12, 6))
    for i in range(NUM_ECOSYSTEMS):
        plt.plot(num_wolves[i], label=f'Wolves Ecosystem {i+1}')
        plt.plot(num_sheeps[i], label=f'Sheep Ecosystem {i+1}')
    plt.xlabel('Time Steps')
    plt.ylabel('Number of Agents')
    plt.title('Population Dynamics of Wolves and Sheep (Multiple Ecosystems)')
    plt.legend()
    plt.show()
    '''


if __name__ == "__main__":
    main_vmap()