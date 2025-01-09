import jax
import jax.numpy as jnp
from flax import struct

from abmax.structs import *



def create_agents(agent:Agent, params:Params, num_agents:jnp.int32, num_active_agents:jnp.int32, agent_type:jnp.int16, key:jax.random.PRNGKey)->Agent:
    """
    Create agents with the given parameters
    assumption: create_agent method of the agent class is defined in the agent class by the user and returns an object of the agent class

    Args:
        agent: Agent class
        params: create parameters of the agent
        num_agents: Number of agents to create
        num_active_agents: Number of active agents
        agent_type: Type of agent
        key: Random key

    Returns:
        agents: object of Agent class, vmapped over params, ids, active_states, subkeys
    
    """
    ids = jnp.arange(num_agents)

    key, *subkeys = jax.random.split(key, num_agents+1)
    subkeys = jnp.array(subkeys)
    
    active_states = jnp.ones(num_agents)
    active_states = jnp.where(ids<num_active_agents, 1, 0)
    
    return jax.vmap(agent.create_agent, in_axes=(None, 0, 0, 0, 0))(agent_type, params, ids, active_states, subkeys)

def create_sets(set:Set, set_params:Params, set_type:jnp.int32,
                agent:Agent, agent_params:Params, agent_type:jnp.int32,
                num_sets:jnp.int32, num_agents:jnp.int32, num_active_agents:jnp.int32, key:jax.random.PRNGKey)->Set:
    """
    Create agent sets with the given parameters
    this is a vmapped version of Set.
    to create just one set, consider creating the set directly in the simulation
    assumption: create_set method of the set class is defined in the set class by the user and returns an object of the set class

    Args:
        set: Set class
        set_params: create parameters of the set (unique for each set, same for all the agents in the set)
        set_type: Type of set
        agent: Agent class
        agent_params: create parameters of the agents (unique for each agent, vmapped twice, once for each set and once for each agent)
        agent_type: Type of agent
        num_set: Number of sets to create
        num_agents: Number of agents to create
        num_active_agents: Number of active agents (different for each set)
        key: Random key


    Returns:
        sets: object of Set class
    
    """
    ids = jnp.arange(num_sets)

    key, *agent_subkeys = jax.random.split(key, num_sets+1)
    agent_subkeys = jnp.array(agent_subkeys)# each set has a different key
    
    key, *set_subkeys = jax.random.split(key, num_sets+1)
    set_subkeys = jnp.array(set_subkeys)
    
    agents = jax.vmap(create_agents, in_axes=(None, 0, None, 0, None, 0))(agent, agent_params, num_agents, num_active_agents, agent_type, agent_subkeys)
    
    return jax.vmap(set.create_set, in_axes=(None, 0, 0, 0, 0, None, 0))(num_agents, num_active_agents, agents, set_params, ids, set_type, set_subkeys)


def step_agents(step_func:callable, step_params:Params, input:Signal, set:Set)->Set:
    """
    Step the agents in the set with the given parameters and input
    assumption: step_func is defined in the agent class by the user and returns an object of the agent class
    Args:
        step_func: function to step the agents,
        step_params: parameters to step the agents, this is same for all the agents in the set
        input: input signal
        set: Set of agents
    returns:
        new set of agents, where the agents are replaced by the new agents
    """
    new_agents = jax.vmap(jax.jit(step_func), in_axes=(0, 0, None))(set.agents, input, step_params)
    return set.replace(agents=new_agents)

jit_step_agents = jax.jit(step_agents, static_argnums=(0,))


def step_sets(step_func:callable, step_params:Params, input:Signal, set:Set)->Set:
    """
    step the set of agents with the given parameters and input
    note: here we are updating the state of the set and not the agents
    think of this like a global parameter in simulation that gets updated for all the agents in the set
    This is also vmapped over the sets, for just one set, consider updating the state of the set directly in the simulation

    assumption: step_func is defined in the set class by the user and returns an object of the set class

    Args:
        step_func: function to step the set
        step_params: parameters to step the set
        input: input signal
        set: Set of agents
    returns:
        new set of agents, where the set_state is replaced by the new set_state

    """
    return jax.vmap(jax.jit(step_func), in_axes=(0, 0, None))(set, input, step_params)

jit_step_sets = jax.jit(step_sets, static_argnums=(0,))



def step_agents_in_sets(step_func:callable, step_params:Params, input:Signal, set:Set)->Set:
    """
    step the agents for all the sets. this is a vmapped version of step_agents for all the worlds in the simulation
    note: here we are updating the agents and not the state of the set
    this is vmapped over the sets, for just one set, consider using the step_agents method
    so 2 vmaps are used here, one for the sets and one for the agents in the set

    assumption: step_func is defined in the agent class by the user and returns an object of the agent class
    Args:
        step_func: function to step the agents
        step_params: parameters to step the agents
        input: input signal
        set: Set of agents
    returns:
        new set of agents, where the agents are replaced by the new agents
    """
    def step_agents(step_params_per_set, input_per_set, set):
        new_agents = jax.vmap(jax.jit(step_func), in_axes=(0, 0, None))(set.agents, input_per_set, step_params_per_set)
        return set.replace(agents=new_agents)
    return jax.vmap(step_agents, in_axes=(0, 0, 0))(step_params, input, set)

jit_step_agents_in_sets = jax.jit(step_agents_in_sets, static_argnums=(0,))
        



def add_agents(add_func:callable, add_params:Params, num_agents_add:jnp.int32, set:Set)->Set:
    """
    Add agents to the set
    assumption: add_func is defined in the agent class by the user and returns an object of the agent class

    args:
        add_func: function to add agents
        num_agents_add: number of agents to add
        add_params: parameters to add agents
        set: Set of agents
    returns:
        new set of agents, where the agents are replaced by the new agents
    """
    id_last_active = set.num_active_agents
    max_agents_add = set.num_agents - id_last_active
    num_agents_add = jnp.minimum(num_agents_add, max_agents_add)

    def add_data(idx, agents):
        new_agent = jax.jit(add_func)(agents, idx, add_params)
        new_agents = jax.tree_util.tree_map(lambda x,y:x.at[idx].set(y), agents, new_agent)
        return new_agents
    
    new_agents = jax.lax.fori_loop(id_last_active, id_last_active+num_agents_add, add_data, set.agents)
    return set.replace(agents=new_agents, num_active_agents = set.num_active_agents + num_agents_add)
jit_add_agents = jax.jit(add_agents, static_argnums=(0,))



def add_agents_in_sets(add_func:callable, add_params:Params, num_agents_add:jnp.int32, set:Set)->Set:
    """
    Add agents to the sets, this is a vmapped version of add_agents for all the worlds in the simulation
    here different number of agents can be added to different sets

    assumption: add_func is defined in the agent class by the user and returns an object of the agent class

    args:
        add_func: function to add agents
        num_agents_add: number of agents to add
        add_params: parameters to add agents
        set: Set of agents
    returns:
        new set of agents, where the agents are replaced by the new agents
    
    """
    def _add_agents(num_agents_add, add_params, set):
        id_last_active = set.num_active_agents
        max_agents_add = set.num_agents - id_last_active
        num_agents_add = jnp.minimum(num_agents_add, max_agents_add)

        def add_data(idx, agents):
            new_agent = jax.jit(add_func)(agents, idx, add_params)
            new_agents = jax.tree_util.tree_map(lambda x,y:x.at[idx].set(y), agents, new_agent)
            return new_agents
        
        new_agents= jax.lax.fori_loop(id_last_active, id_last_active+num_agents_add, add_data, set.agents)
    
        return set.replace(agents=new_agents, num_active_agents=set.num_active_agents+num_agents_add)
    
    return jax.vmap(_add_agents)(num_agents_add, add_params, set)
jit_add_agents_in_sets = jax.jit(add_agents_in_sets, static_argnums=(0,))


def remove_agents(remove_func:callable, remove_params:Params, num_agents_remove:jnp.int32, set:Set)->tuple:
    """
    Remove agents from the set
    assumption: remove_func is defined in the agent class by the user and returns an object of the agent class

    args:
        remove_func: function to remove agents
        num_agents_remove: number of agents to remove
        remove_params: parameters to remove agents, SHOULD HAVE 'remove_indx' key in the content, The max remove_id is less than the number of agents active
        which contains the ids of the agents to remove
        set: Set of agents
    returns:
        a tuple of:
        new set of agents, where the agents are replaced by the new agents
        sorted_indx: indexes of the agents in the sorted order (descending order of active_state)
    """
    num_agents_remove = jnp.minimum(num_agents_remove, set.num_active_agents)

    def remove_data(idx, agents):
        remove_indx = remove_params.content['remove_indx']
        new_agent = jax.jit(remove_func)(agents, remove_indx[idx], remove_params)
        new_agents = jax.tree_util.tree_map(lambda x,y:x.at[remove_indx[idx]].set(y), agents, new_agent)
        return new_agents
    
    new_agents = jax.lax.fori_loop(0, num_agents_remove, remove_data, set.agents)
    new_agents, sorted_indx = jit_sort_agents(-1*new_agents.active_state, new_agents)
    return set.replace(agents=new_agents, num_active_agents = set.num_active_agents - num_agents_remove), sorted_indx

jit_remove_agents = jax.jit(remove_agents, static_argnums=(0,))


def remove_agents_in_sets(remove_func:callable, remove_params:Params, num_agents_remove:jnp.int32, set:Set)->tuple:
    """
    Remove particular agents from the set
    assumption: remove_func is defined in the agent class by the user and returns an object of the agent class
    here we remove different number of agents from different sets

    args:
        remove_func: function to remove agents
        num_agents_remove: number of agents to remove
        remove_params: parameters to remove agents, SHOULD HAVE 'remove_indx' key in the content which
        contains the ids of the agents to remove
        set: Set of agents
    returns:
        a tuple of:
        new set of agents, where the agents are replaced by the new agents
        sorted_indx: indexes of the agents in the sorted order (descending order of active_state)
    """
    def _remove_agents(num_agents_remove, remove_params, set):
        num_agents_remove = jnp.minimum(num_agents_remove, set.num_active_agents)

        def remove_data(idx, agents):
            remove_indx = remove_params.content['remove_indx']
            new_agent = jax.jit(remove_func)(agents, remove_indx[idx], remove_params)
            new_agents = jax.tree_util.tree_map(lambda x,y:x.at[remove_indx[idx]].set(y), agents, new_agent)
            return new_agents
        
        new_agents = jax.lax.fori_loop(0, num_agents_remove, remove_data, set.agents)
        new_agents, sorted_indx = jit_sort_agents(-1*new_agents.active_state, new_agents)
        return set.replace(agents=new_agents, num_active_agents=set.num_active_agents-num_agents_remove), sorted_indx
    
    return jax.vmap(_remove_agents)(num_agents_remove, remove_params, set)

jit_remove_agents_in_sets = jax.jit(remove_agents_in_sets, static_argnums=(0,))


def set_agents(set_func:callable, set_params:Params, num_agents_set:jnp.int32, set:Set)->Set:
    """
    Set particular agents in the set
    assumption: set_func is defined in the agent class by the user and returns an object of the agent class

    args:
        set_func: function to set agents
        num_agents_set: number of agents to set
        set_params: parameters to set agents, SHOULD HAVE 'set_indx' key in the content which
        contains the ids of the agents to set
        set: Set of agents

    returns:
        new set of agents, where the agents are replaced by the new agents
    
    """
    num_agents_set = jnp.minimum(num_agents_set, set.num_active_agents)
    def set_data(idx, agents):
        set_indx = set_params.content['set_indx']
        new_agent = jax.jit(set_func)(agents, set_indx[idx], set_params)
        new_agents = jax.tree_util.tree_map(lambda x,y:x.at[set_indx[idx]].set(y), agents, new_agent)
        return new_agents
    
    new_agents = jax.lax.fori_loop(0, num_agents_set, set_data, set.agents)
    return set.replace(agents=new_agents)

jit_set_agents = jax.jit(set_agents, static_argnums=(0,))


def set_agents_in_set(set_func:callable, set_params:Params, num_agents_set:jnp.int32, set:Set)->Set:
    """
    Set particular agents in the set
    assumption: set_func is defined in the agent class by the user and returns an object of the agent class
    here we set different number of agents in different sets

    args:
        set_func: function to set agents
        num_agents_set: number of agents to set
        set_params: parameters to set agents, SHOULD HAVE 'set_indx' key in the content which
        contains the ids of the agents to set
        set: Set of agents

    returns:
        new set of agents, where the agents are replaced by the new agents
    """
    def _set_agents(num_agents_set, set_params, set):
        num_agents_set = jnp.minimum(num_agents_set, set.num_active_agents)
        def set_data(idx, agents):
            set_indx = set_params.content['set_indx']
            new_agent = jax.jit(set_func)(agents, set_indx[idx], set_params)
            new_agents = jax.tree_util.tree_map(lambda x,y:x.at[set_indx[idx]].set(y), agents, new_agent)
            return new_agents
        
        new_agents = jax.lax.fori_loop(0, num_agents_set, set_data, set.agents)
        return set.replace(agents=new_agents)
    return jax.vmap(_set_agents)(num_agents_set, set_params, set)
jit_set_agents_in_set = jax.jit(set_agents_in_set, static_argnums=(0,))


def sort_agents(quantity:jnp.array, agents:Agent)->Agent:
    """
    Sort the agents based on the quantity
    assumption: quantity is a 1D array of the same length as the number of agents

    args:
        quantity: quantity to sort the agents( multiply by -1 if you want to sort in descending order)
        agents: agents
    returns:
        new agents which are sorted based on the quantity
        sorted_indx: indexes of the agents in the ascending order
            for example consider quantity = [3, 1, 2]
            then the sorted_indx = [1, 2, 0] as 1 < 2 < 3 
    """
    quantity = jnp.reshape(quantity, (-1,))
    sorted_indx = jnp.argsort(quantity)
    new_agents = jax.tree_util.tree_map(lambda x: jnp.take(x, sorted_indx, axis=0), agents)
    return new_agents, sorted_indx

jit_sort_agents = jax.jit(sort_agents)

def sort_sets(quantity:jnp.array, sets:Set)->Set:
    """
    Sort the agents in the set based on the quantity
    assumption: quantity is a 1D array of the same length as the number of agents

    args:
        quantity: quantity to sort the agents
        set: Set of agents
    returns:
        new set of agents which are sorted based on the quantity
        sorted_indx: indexes of the agents in the ascending order
            for example consider quantity = [3, 1, 2]
            then the sorted_indx = [1, 2, 0] as 1 < 2 < 3

    """
    quantity = jnp.reshape(quantity, (-1,))
    sorted_indx = jnp.argsort(quantity)
    new_sets = jax.tree_util.tree_map(lambda x: jnp.take(x, sorted_indx, axis=0), sets)
    return new_sets, sorted_indx

jit_sort_sets = jax.jit(sort_sets)

def select_agents(select_func:bool, select_params:Params, set:Set)->tuple:
    """
    Select agents based on the select function
    assumption: select_func is defined by the user and 
    it takes in agents and select_params and returns a boolean array of the same length as the number of agents
    args:
        select_func: function to select agents
        select_params: parameters to select agents
        set: Set of agents
    returns:
        a tuple of:
        selected_indx_len: number of agents selected
        sort_selected_indx: indexes of the agents in the sorted order (descending order of active_state)
        So these 2 values can be used in a for loop to iterate over the selected agents
    """
    selected_indx = jnp.where(jax.jit(select_func)(set.agents, select_params), 1.0, 0.0)
    selected_indx = jnp.reshape(selected_indx,(-1,))

    sort_selected_indx = jnp.argsort(-1*selected_indx) # logic: 1 is greater than 0, so everywhere there is 1 will bubble up    
    selected_indx_len = jnp.sum(selected_indx, dtype=jnp.int32) # logic: sum of 1s is the number of selected agents
    return selected_indx_len, sort_selected_indx

jit_select_agents = jax.jit(select_agents, static_argnums=(0,))



def select_agents_in_sets(select_func:bool, select_params:Params, set:Set)->tuple:
    """
    Select agents based on the select function
    this is a vmapped version of select_agents for all the worlds in the simulation
    here different number of agents can be selected in different sets

    assumption: select_func is defined by the user and
    it takes in agents and select_params and returns a boolean array of the same length as the number of agents

    args:
        select_func: function to select agents
        select_params: parameters to select agents
        set: Set of agents
    returns:
        a tuple of:
        selected_indx_len: number of agents selected
        sort_selected_indx: indexes of the agents in the sorted order (descending order of active_state)
        So these 2 values can be used in a for loop to iterate over the selected agents

    """

    def _select_agents(select_params, agents):
        selected_indx = jnp.where(jax.jit(select_func)(agents, select_params), 1.0, 0.0)
        selected_indx = jnp.reshape(selected_indx,(-1,))

        sort_selected_indx = jnp.argsort(-1*selected_indx)    
        selected_indx_len = jnp.sum(selected_indx, dtype=jnp.int32)
        
        return selected_indx_len, sort_selected_indx
    
    return jax.vmap(_select_agents)(select_params, set.agents)

jit_select_agents_in_sets = jax.jit(select_agents_in_sets, static_argnums=(0,))

def select_sets(select_func:bool, select_params:Params, set:Set)->tuple:
    """
    Select sets based on the select function
    assumption: select_func is defined by the user and
    it takes in agents and select_params and returns a boolean array of the same length as the number of sets

    args:
        select_func: function to select sets
        select_params: parameters to select sets
        set: Set of agents
    returns:
        a tuple of:
        selected_indx_len: number of sets selected
        sort_selected_indx: indexes of the sets in the sorted order (descending order of active_state)
        So these 2 values can be used in a for loop to iterate over the selected sets
    """
    selected_indx = jnp.where(jax.jit(select_func)(set, select_params), 1.0, 0.0)
    selected_indx = jnp.reshape(selected_indx,(-1,))
    sort_selected_indx = jnp.argsort(-1*selected_indx)
    selected_indx_len = jnp.sum(selected_indx, dtype=jnp.int32)
    return selected_indx_len, sort_selected_indx

jit_select_sets = jax.jit(select_sets, static_argnums=(0,))