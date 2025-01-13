from abmax.structs import *
from abmax.functions import *
import jax.numpy as jnp
import jax.random as random
import jax
import time

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
            
            energy = jax.random.randint(create_keys[2], minval=1, maxval=2*params.content['delta_energy'], shape=(1,))

            state_content = {'X_pos': X_pos, 'Y_pos': Y_pos, 'energy': energy, 'reproduce': 0}
            state = State(content=state_content)
            return state
        
        def create_inactive_agent():
            state_content = {'X_pos': jnp.array([-1]), 'Y_pos': jnp.array([-1]), 'energy': jnp.array([-1]), 'reproduce': 0}
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

            energy_new = energy - 1 + agent.params.content['delta_energy']*energy_in

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
    def remove_agent(agents, idx, remove_params):
        agent_to_remove = jax.tree_util.tree_map(lambda x: x[idx], agents)
        state_content = {'X_pos': jnp.array([-1]), 'Y_pos': jnp.array([-1]), 'energy': jnp.array([-1]), 'reproduce': 0} 
        state = State(content=state_content)
        age = 0.0
        active_state = False
        return agent_to_remove.replace(state=state, age=age, active_state=active_state)
    
    @staticmethod
    def add_agent(agents, idx, add_params):
        
        copy_indx = add_params.content['copy_indx'] 
        # copy_ids contains the ids of the agents that are selected for reproduction
        # we need this to know where to spawn the new agent


        num_active_agents = add_params.content['num_active_agents']

        agent_to_add = jax.tree_util.tree_map(lambda x: x[idx], agents)
        agent_to_copy = jax.tree_util.tree_map(lambda x: x[copy_indx[idx - num_active_agents]], agents)
        
        # copying the position but halving the energy of the agent to copy
        X_pos = agent_to_copy.state.content['X_pos']
        Y_pos = agent_to_copy.state.content['Y_pos']
        energy = agent_to_copy.state.content['energy']/2

        state_content = {'X_pos': X_pos, 'Y_pos': Y_pos, 'energy': energy, 'reproduce': 0}
        state = State(content=state_content)
        age = 0.0
        active_state = True
        return agent_to_add.replace(state=state, age=age, active_state=active_state)
    
    @staticmethod
    def set_agent(agents, idx, set_params):
        """
        This function is used to set energy of all the animals that have reproduced to half
        and the reproduce flag to 0
        """
        agent_to_set = jax.tree_util.tree_map(lambda x: x[idx], agents)
        X_pos = agent_to_set.state.content['X_pos']
        Y_pos = agent_to_set.state.content['Y_pos']
        energy = agent_to_set.state.content['energy']/2
        
        state_content = {'X_pos': X_pos, 'Y_pos': Y_pos, 'energy': energy, 'reproduce': 0}
        state = State(content=state_content)
        
        return agent_to_set.replace(state=state)
    

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
    num_step: jnp.int32

    @staticmethod
    def create_ecossystem(grass_regrowth_time, wolf_reproduction_probab, wolf_energy, init_wolves, sheep_reproduction_probab, 
                 sheep_energy, init_sheeps, space_size, sim_steps, key):

        key, grass_key, grass_set_key = jax.random.split(key, 3)
        num_grass_agents = int(space_size**2)
        grass_params = Params(content={'regrowth_time': jnp.tile(grass_regrowth_time, num_grass_agents), 
                               'x_max': jnp.tile(space_size, num_grass_agents)})
        
        grass_agent = create_agents(Grass, params = grass_params, num_agents = num_grass_agents, 
                                    num_active_agents = num_grass_agents, agent_type=1, key=grass_key)
        
        grass_set = Set(agents=grass_agent, num_agents=num_grass_agents, state=None, params=None, policy=None, id=0,
                        num_active_agents=num_grass_agents, key=grass_set_key, set_type=1)
        
        key, wolf_key, wolf_set_key = jax.random.split(key, 3)
        num_active_wolves = init_wolves
        num_max_wolves = int(1.2*init_wolves)
        wolf_params = Params(content={'reproduction_probab': jnp.tile(wolf_reproduction_probab, num_max_wolves), 
                                      'delta_energy': jnp.tile(wolf_energy, num_max_wolves), 
                                      'x_max': jnp.tile(space_size, num_max_wolves), 
                                      'y_max': jnp.tile(space_size, num_max_wolves)})
        
        wolf_agent = create_agents(Animal, params = wolf_params, num_agents = num_max_wolves,
                                      num_active_agents = num_active_wolves, agent_type=2, key=wolf_key)
        
        wolf_set = Set(agents=wolf_agent, num_agents=num_max_wolves, state=None, params=None, policy=None, id=0,
                        num_active_agents=num_active_wolves, key=wolf_set_key, set_type=2)
        
        key, sheep_key, sheep_set_key = jax.random.split(key, 3)
        num_active_sheeps = init_sheeps
        num_max_sheeps = int(1.2*init_sheeps)
        sheep_params = Params(content={'reproduction_probab': jnp.tile(sheep_reproduction_probab, num_max_sheeps), 
                                       'delta_energy': jnp.tile(sheep_energy, num_max_sheeps), 
                                       'x_max': jnp.tile(space_size, num_max_sheeps), 
                                       'y_max': jnp.tile(space_size, num_max_sheeps)})
        
        sheep_agent = create_agents(Animal, params = sheep_params, num_agents = num_max_sheeps,
                                        num_active_agents = num_active_sheeps, agent_type=3, key=sheep_key)
        
        sheep_set = Set(agents=sheep_agent, num_agents=num_max_sheeps, state=None, params=None, policy=None, id=0,
                        num_active_agents=num_active_sheeps, key=sheep_set_key, set_type=3)
        
        return Ecosystem(wolf_set=wolf_set, sheep_set=sheep_set, grass_set=grass_set, num_step=sim_steps)
    
    @staticmethod
    def select_dead_sheeps(sheeps:Agent, select_params:Params):
        energy = jnp.reshape(sheeps.state.content['energy'], (-1))
        is_eaten = jnp.reshape(select_params.content['sheeps_eaten'], (-1))
        is_dead = jnp.logical_or(energy <= 0, is_eaten)
        is_dead = jnp.logical_and(is_dead, sheeps.active_state)
        return is_dead
    
    @staticmethod
    def select_dead_wolves(wolves:Agent, select_params:Params):
        energy = jnp.reshape(wolves.state.content['energy'], (-1))
        is_dead = jnp.logical_and(energy <= 0, wolves.active_state)
        return is_dead
    
    @staticmethod
    def select_reproduce_animals(animals:Agent, select_params:Params):
        is_reproduce = jnp.reshape(animals.state.content['reproduce'], (-1))
        return jnp.logical_and(is_reproduce, animals.active_state)
    
    @staticmethod
    @jax.jit
    def add_animals(animal_set:Set):
        # select animals that are reproducing
        num_animals_reproduce, reproduce_indx = jit_select_agents(select_func = Ecosystem.select_reproduce_animals,
                                                                    select_params = None, set = animal_set)
        # add new animals
        old_active_agents = animal_set.num_active_agents
        animal_add_params_content = {'copy_indx': reproduce_indx, 'num_active_agents': num_animals_reproduce}
        animal_add_params = Params(content=animal_add_params_content)
        animal_set = jit_add_agents(add_func = Animal.add_agent, add_params = animal_add_params, 
                                        num_agents_add = num_animals_reproduce, set = animal_set)
        
        # set the reproduce flag to 0 and energy to half for the animals that have reproduced
        # Note: All animals selected for reproduction may not have reproduced due to max number of animals constraint
        num_animals_repr = animal_set.num_active_agents - old_active_agents
        animal_set_params_content = {'set_indx': reproduce_indx} # This is required internally so dont change the name
        animal_set_params = Params(content=animal_set_params_content)
        animal_set = jit_set_agents(set_func = Animal.set_agent, set_params = animal_set_params, 
                                        num_agents_set = num_animals_repr, set = animal_set)
        
        return animal_set


    @staticmethod
    def step(ecosystem, t):
        wolves_energy_in, sheeps_energy_in, sheeps_eaten, grasses_eaten = jit_interaction(ecosystem.wolf_set.agents, 
                                                                                          ecosystem.sheep_set.agents, 
                                                                                          ecosystem.grass_set.agents)
        
        grass_step_signal = Signal(content={'energy_out': grasses_eaten})
        new_grass_set = jit_step_agents(step_func = Grass.step_agent, step_params = None, input = grass_step_signal,
                                              set = ecosystem.grass_set)
        
        sheep_step_signal = Signal(content={'energy_in': sheeps_energy_in})
        new_sheep_set = jit_step_agents(step_func = Animal.step_agent, step_params = None, input = sheep_step_signal,
                                              set = ecosystem.sheep_set)
        
        wolf_step_signal = Signal(content={'energy_in': wolves_energy_in})
        new_wolf_set = jit_step_agents(step_func = Animal.step_agent, step_params = None, input = wolf_step_signal,
                                              set = ecosystem.wolf_set)
        
        # remove dead sheep
        dead_sheep_select_params_content = {'sheeps_eaten': sheeps_eaten}
        dead_sheep_select_params = Params(content=dead_sheep_select_params_content)
        num_dead_sheeps, sheep_remove_indx = jit_select_agents(select_func = Ecosystem.select_dead_sheeps,
                                                                select_params = dead_sheep_select_params, 
                                                                set = new_sheep_set)
        dead_sheep_remove_params_content = {'remove_indx': sheep_remove_indx} # This is required internally so dont change the name

        dead_sheep_remove_params = Params(content=dead_sheep_remove_params_content)
        new_sheep_set, _ = jit_remove_agents(remove_func = Animal.remove_agent, 
                                          remove_params = dead_sheep_remove_params, num_agents_remove = num_dead_sheeps,
                                          set = new_sheep_set)
        
        # remove dead wolves
        num_dead_wolves, wolf_remove_indx = jit_select_agents(select_func = Ecosystem.select_dead_wolves,
                                                             select_params = None, set = new_wolf_set)
        dead_wolf_remove_params_content = {'remove_indx': wolf_remove_indx}
        dead_wolf_remove_params = Params(content=dead_wolf_remove_params_content)
        new_wolf_set, _ = jit_remove_agents(remove_func = Animal.remove_agent, 
                                         remove_params = dead_wolf_remove_params, num_agents_remove = num_dead_wolves,
                                         set = new_wolf_set)
        
        # add new animals
        new_sheep_set = Ecosystem.add_animals(new_sheep_set)
        new_wolf_set = Ecosystem.add_animals(new_wolf_set)
        
        return ecosystem.replace(grass_set=new_grass_set, sheep_set=new_sheep_set, wolf_set=new_wolf_set), (ecosystem.sheep_set.num_active_agents, ecosystem.wolf_set.num_active_agents)

    @staticmethod
    @jax.jit
    def run_loop(ecosystem, ts):
        ecosystem, num_agents = jax.lax.scan(Ecosystem.step, ecosystem, ts)
        return ecosystem, num_agents
    
    @staticmethod
    @jax.jit
    def run_loop_vmap(ecosystems, ts):
        ecosystems, num_agents = jax.lax.scan( jax.vmap(Ecosystem.step, in_axes=(0, None)), ecosystems, ts)
        return ecosystems, num_agents
    
    @staticmethod
    def run(ecosystem):
        ts = jnp.arange(ecosystem.num_step)
        ecosystem, num_agents = Ecosystem.run_loop(ecosystem, ts)
        return ecosystem, num_agents
    
    @staticmethod
    def run_vmap(ecosystems):
        #print("creating ts")
        ts = jnp.arange(ecosystems.num_step[0])
        #print("starting sim")
        #time_begin = time.time()
        ecosystems, num_agents = Ecosystem.run_loop_vmap(ecosystems, ts)
        #time_end = time.time()
        #print("Time taken for 100:", time_end - time_begin)
        return ecosystems, num_agents



def main(grass_regrowth_time = 40,
         space_size = 1000,
         wolf_reproduction_probab = 0.5,
         wolf_energy = 40,
         init_wolves = 5000,
         sheep_reproduction_probab = 0.2,
         sheep_energy = 10,
         init_sheeps = 10000,
         sim_steps = 100,
         key = random.PRNGKey(0)):
    
    ecosystem = Ecosystem.create_ecossystem(grass_regrowth_time, wolf_reproduction_probab, wolf_energy, init_wolves, 
                                            sheep_reproduction_probab, sheep_energy, init_sheeps, space_size, sim_steps, key)

    ecosystem, num_agents = Ecosystem.run(ecosystem)
    return num_agents

def main_vmap(grass_regrowth_time = 20,
         space_size = 100,
         wolf_reproduction_probab = 0.2,

         wolf_energy = 20,
         init_wolves = 500,
         sheep_reproduction_probab = 0.4,

         sheep_energy = 10,
         init_sheeps = 1000,
         sim_steps = 100,

         key = random.PRNGKey(0)):
    key, *ecosystem_keys = random.split(key, 5)
    ecosystem_keys = jnp.array(ecosystem_keys)
    
    print("creating ecosystems")
    ecosystems = jax.vmap(Ecosystem.create_ecossystem, in_axes=(None, None, None, None, None, None, None, None, None, 0))(grass_regrowth_time, wolf_reproduction_probab, wolf_energy, init_wolves, sheep_reproduction_probab, sheep_energy, init_sheeps, space_size, sim_steps, ecosystem_keys)

    ecosystems, num_agents = Ecosystem.run_vmap(ecosystems)

    return num_agents




if __name__ == "__main__":
    num_agents = main()
    #num_agents = main_vmap()
    print(num_agents)


# all ecosystems created after this are used for benchmarking.

grass_regrowth_time = 10
space_size = 100
wolf_reproduction_probab = 0.2
wolf_energy = 40
init_wolves = 500
sheep_reproduction_probab = 0.4
sheep_energy = 10
init_sheeps = 1000
sim_steps = 100
key = random.PRNGKey(0)
    
ecosystem_small = Ecosystem.create_ecossystem(grass_regrowth_time, wolf_reproduction_probab, wolf_energy, init_wolves, 
                                        sheep_reproduction_probab, sheep_energy, init_sheeps, space_size, sim_steps, key)

grass_regrowth_time = 40
space_size = 1000
wolf_reproduction_probab = 0.2
wolf_energy = 13
init_wolves = 5000
sheep_reproduction_probab = 0.2
sheep_energy = 5
init_sheeps = 10000
sim_steps = 100
key = random.PRNGKey(0)
    
ecosystem_large = Ecosystem.create_ecossystem(grass_regrowth_time, wolf_reproduction_probab, wolf_energy, init_wolves, 
                                        sheep_reproduction_probab, sheep_energy, init_sheeps, space_size, sim_steps, key)   
        

grass_regrowth_time = 20
space_size = 100
wolf_reproduction_probab = 0.2
wolf_energy = 20
init_wolves = 500
sheep_reproduction_probab = 0.4
sheep_energy = 10
init_sheeps = 1000
sim_steps = 100

key = random.PRNGKey(0)
key, *ecosystem_keys = random.split(key, 501)
ecosystem_keys = jnp.array(ecosystem_keys)
ecosystem_vmap = jax.vmap(Ecosystem.create_ecossystem, in_axes=(None, None, None, None, None, None, None, None, None, 0))(grass_regrowth_time, wolf_reproduction_probab, wolf_energy, init_wolves, sheep_reproduction_probab, sheep_energy, init_sheeps, space_size, sim_steps, ecosystem_keys)
