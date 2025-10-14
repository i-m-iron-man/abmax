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


def set_agents_rank_match(set_func:callable, set_params:Params, mask_params:Params, num_agents:jnp.int32, set:Set)->Set:
    """
    Set particular agents in the set based on the rank match algorithm
    Make sure that the len(change_mask) == len(changes) in set_params.content
    assumption: set_func is defined in the agent class by the user and returns an object of the agent class

    args:
        set_func: function to set agents
        set_params: parameters to set agents, SHOULD HAVE 'set_indx' key in the content which
        contains the ids of the agents to set
        mask_params: parameters to set agents, SHOULD HAVE 'select_mask' and 'change_mask' keys in the content
        num_agents: number of agents in the set, this is used to limit the number of agents that can be set
        set: Set of agents
    returns:
        new set of agents, where the agents are replaced by the new agents
        num_changes: number of changes made, this is the number of agents that were set
    """
    select_mask = mask_params.content['select_mask']
    change_mask = mask_params.content['change_mask']
    

    cumsum_select_mask = jnp.cumsum(select_mask, dtype=jnp.int32)
    unique_select_mask = jnp.multiply(cumsum_select_mask, select_mask)

    cumsum_change_mask = jnp.cumsum(change_mask, dtype=jnp.int32)
    unique_change_mask = jnp.multiply(cumsum_change_mask, change_mask)

    num_changes = jnp.minimum(cumsum_select_mask[-1], cumsum_change_mask[-1]) # this is the number of changes that can be made

    def check_num_agent():
        return jnp.where(unique_change_mask > num_agents, 0, unique_change_mask), jnp.minimum(num_changes, num_agents)
    unique_change_mask, num_changes = jax.lax.cond(num_agents>-1, lambda _: check_num_agent(), lambda _: (unique_change_mask, num_changes), None)

    def set_agent(agent, select_mask_el):
        
        def check_mask(set_param, change_mask_el):
            def if_match():
                return jax.jit(set_func)(agent, set_param), 1
            def if_not_match():
                return agent, 0
            cond = jnp.logical_and(change_mask_el==select_mask_el, change_mask_el!=0)
            return jax.lax.cond(cond, lambda _: if_match(), lambda _: if_not_match(), None)
        
        agent_candidates, update_mask = jax.vmap(check_mask)(set_params, unique_change_mask)
        is_updated = jnp.sum(update_mask, dtype=jnp.int32) > 0
        update_index = jnp.argmax(update_mask)
        return jax.lax.cond(is_updated, lambda _: jax.tree_util.tree_map(lambda x: x[update_index], agent_candidates), lambda _: agent, None)

    new_agents = jax.vmap(set_agent)(set.agents, unique_select_mask)
    num_active_agents = jnp.sum(new_agents.active_state, dtype=jnp.int32)
    return set.replace(agents=new_agents, num_active_agents=num_active_agents), num_changes
jit_set_agents_rank_match = jax.jit(set_agents_rank_match, static_argnums=(0,))


def set_agents_mask(set_func:callable, set_params:Params, mask_params:Params, num_agents:jnp.int32, set:Set)->Set:
    """
    this function updates the agents when the changes are not unique, for eg removing agents or stepping agents
    mask is used to select the agents that need to be set
    assumption: set_func is defined in the agent class by the user and returns an object of the agent class
    args:
        set_func: function to set agents
        set_params: parameters to set agents
        mask_params: parameters to set agents, SHOULD HAVE 'set_mask' key in the content, highlighting the agents that need to be set
        num_agents: number of agents in the set, this is used to limit the number of agents that can be set
        set: Set of agents
    returns:
        new set of agents, where the agents are replaced by the new agents
        num_active_agents: number of active agents in the set after setting the agents
    """
    
    set_mask = mask_params.content['set_mask']
    unique_set_mask = jnp.multiply(jnp.cumsum(set_mask, dtype=jnp.int32), set_mask)

    unique_set_mask = jax.lax.cond(num_agents>-1,
        lambda _: jnp.where(unique_set_mask > num_agents, 0, unique_set_mask), 
        lambda _: unique_set_mask, 
        None
    )
    
    def set_agent(agent, set_mask_el, set_param):
        def if_set():
            return jax.jit(set_func)(agent, set_param)
        def if_not_set():
            return agent
        return jax.lax.cond(set_mask_el>0, lambda _: if_set(), lambda _: if_not_set(), None)

    new_agents = jax.vmap(set_agent, in_axes=(0, 0, None))(set.agents, unique_set_mask, set_params)
    num_active_agents = jnp.sum(new_agents.active_state, dtype=jnp.int32)
    return set.replace(agents=new_agents, num_active_agents=num_active_agents)

jit_set_agents_mask = jax.jit(set_agents_mask, static_argnums=(0,))



def set_agents_sci(set_func:callable, set_params:Params, num_agents_set:jnp.int32, set:Set)->Set:
    """
    apply unique updates to agents following the sort-count-iterate algorithm
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
    def set_data(idx, agents):
        set_indx = set_params.content['set_indx']
        new_agent = jax.jit(set_func)(agents, idx, set_params)
        new_agents = jax.tree_util.tree_map(lambda x,y:x.at[set_indx[idx]].set(y), agents, new_agent)
        return new_agents

    new_agents = jax.lax.fori_loop(0, num_agents_set, set_data, set.agents)
    return set.replace(agents=new_agents)
jit_set_agents_sci = jax.jit(set_agents_sci, static_argnums=(0,))


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