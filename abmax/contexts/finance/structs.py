from abmax.structs import *
from abmax.functions import *

import jax.numpy as jnp
import jax.random as random
import jax
from flax import struct

NUM_LOBS = 10

@struct.dataclass
class Order(Agent):
    @staticmethod
    def create_agent(type: jnp.int32, params: Params, id: jnp.int32, active_state: bool, key: jax.random.PRNGKey) -> Agent:
        '''
        Create a blank order agent, assumption: active_state is false

        args:
            type: the type of the agent
            params: parameters to create the agent, should contain polarity
            id: the id of the agent
            active_state: whether the agent is active or not
            key: random key for agent creation, not used here
        returns:
            a blank order agent with the given type, params, id, and active_state
        '''
        num_shares = jnp.array([0])  # number of shares remaining in the order # dont put -1 as cumsum will not work
        remove_flag = jnp.array([0]) # flag to indicate if the order is removed
        agent_state_content = {"num_shares": num_shares, "remove_flag": remove_flag}
        agent_state = State(content=agent_state_content) 

        
        polarity = params.content["polarity"] #params.content["polarity"] # jnp.array([0]) -> buy, jnp.array([1]) ->sell
        min_order_price = params.content["min_order_price"]
        max_order_price = params.content["max_order_price"]
        price = jax.lax.cond(polarity[0] == 0, lambda _: jnp.array([min_order_price]), lambda _: jnp.array([max_order_price]), None) # blank_price, 0 for buy, 1000 for sell, makes it easy to sort orders
        trader_id = jnp.array([-1]) # id of the trader who placed the order, -1 for inactive orders

        agent_params = Params(content = {"polarity": polarity, "price": price, "trader_id": trader_id})

        agent = Order(agent_type=type, params=agent_params, id=id, active_state=active_state, state=agent_state, policy=None, key=key, age = 0.0)
        return agent
        

    @staticmethod
    def step_agent(agent: Agent, input: Signal, step_params: Params) -> Agent:
        '''
        deplete the number of shares remaining in the order by the input amount
        If the number of shares remaining is 0, set the remove_flag to 1, indicating the order is ready to be removed.
        Does nothing if the order is inactive

        args:
            agent: the order agent to be stepped
            input: the input signal containing the number of shares to remove
            step_params: parameters for the step, should contain dt (time step)
        returns:
            the updated order agent 

        '''
        def step_active_agent():
            num_shares_remove = input.content["num_shares_remove"]
            num_shares = agent.state.content["num_shares"]
            
            num_shares = jnp.maximum(num_shares - num_shares_remove, 0)
            remove_flag = jax.lax.cond(num_shares[0] == 0, lambda _: jnp.array([1]), lambda _: jnp.array([0]), None)
            agent_state_content = {"num_shares": num_shares, "remove_flag": remove_flag}
            agent_state = State(content=agent_state_content)

            dt = step_params.content["dt"]
            return agent.replace(state=agent_state, age=agent.age + dt)
        
        def step_inactive_agent():
            # Inactive agents do not change state, just return the agent as is
            return agent
        
        return jax.lax.cond(agent.active_state, lambda _: step_active_agent(), lambda _: step_inactive_agent(), None)


    
    @staticmethod
    def remove_agent(agent:Agent, remove_params:Params)->Agent:
        '''
        remove an agent by replacing it with a blank order agent based on its polarity
        arguments:
            agent: the agent to be removed
            remove_params: information about the agent to be removed, not used here
        returns:
            a blank order agent with the same polarity as the removed agent
        '''
        num_shares = jnp.array([0]) # don't put -1 as cumsum will not work
        remove_flag = jnp.array([0]) 
        agent_state_content = {"num_shares": num_shares, "remove_flag": remove_flag}
        agent_state = State(content=agent_state_content)

        polarity = agent.params.content["polarity"]
        min_order_price = remove_params.content["min_order_price"]
        max_order_price = remove_params.content["max_order_price"]
        price = jax.lax.cond(polarity[0] == 0, lambda _: jnp.array([min_order_price]), lambda _: jnp.array([max_order_price]), None)
        trader_id = jnp.array([-1]) # id of the trader who placed the order, -1 for inactive orders
        agent_params = Params(content = {"polarity": polarity, "price": price, "trader_id": trader_id})

        return agent.replace(state=agent_state, params=agent_params, active_state=0, age=0.0)

    @staticmethod
    def add_agent(agents, idx, add_params):
        '''
        add a new order at idx with the given params
        arguments:
            agents: the agents to which the new order will be added
            idx: the index at which the new order will be added
            add_params: parameters for the new order, should contain polarity, price, trader_id, num_shares
        returns:
            a new order agent with the given params at the given index
        '''
        order_to_add = jax.tree_util.tree_map(lambda x: x[idx], agents)
        
        num_active_agents = add_params.content["num_active_agents"]
        param_idx = idx - num_active_agents

        trader_id = add_params.content["trader_id_list"][param_idx] # which trader has placed the order
        price = add_params.content["price_list"][param_idx] # whats the price of the order
        num_shares = add_params.content["num_shares_list"][param_idx] # how many shares in the order
        
        polarity = order_to_add.params.content["polarity"]
        
        remove_flag = jnp.array([0])
        agent_state = State(content={"num_shares": num_shares, "remove_flag": remove_flag})

        agent_params = Params(content = {"polarity": polarity, "price": price, "trader_id": trader_id})

        return order_to_add.replace(state=agent_state, params=agent_params, active_state=1, age=0.0)
    
    @staticmethod
    def add_agent_sci(agents, idx, add_params):
        set_indx = add_params.content["set_indx"]
        
        order_to_add = jax.tree_util.tree_map(lambda x: x[set_indx[idx]], agents)
        
        trader_id = add_params.content["trader_id_list"][idx] # which trader has placed the order
        price = add_params.content["price_list"][idx] # whats the price of the order
        num_shares = add_params.content["num_shares_list"][idx] # how many shares in the order
        
        polarity = order_to_add.params.content["polarity"]
        
        remove_flag = jnp.array([0])
        agent_state = State(content={"num_shares": num_shares, "remove_flag": remove_flag})

        agent_params = Params(content = {"polarity": polarity, "price": price, "trader_id": trader_id})

        return order_to_add.replace(state=agent_state, params=agent_params, active_state=1, age=0.0)
    


@struct.dataclass
class Trader(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
        key, *create_keys = random.split(key, 4)
        
        policy = None
        num_lobs = NUM_LOBS # this is not part of params since shapes depend on the number of lobs
        starting_price = params.content["starting_price"] # all lobs start at the same price
        belief_span = params.content["belief_span"]
        lower_price = starting_price * (1.0 - belief_span)
        upper_price = starting_price * (1.0 + belief_span)

        beliefs = jax.random.uniform(key=create_keys[0], shape=(num_lobs,), minval=lower_price, maxval=upper_price)
        cash = jax.random.uniform(key=create_keys[1], shape=(1,), minval=35000.0, maxval=450000.0)  # cash is a random value between 35000 and 45000
        shares = jax.random.randint(key=create_keys[2], shape=(num_lobs,), minval=100, maxval=1000)  # shares is a random value between 100 and 1000
        
        buy_flag = jnp.tile(False, (num_lobs,))
        buy_num_shares = jnp.tile(-1, (num_lobs,))
        buy_price = jnp.tile(0.0, (num_lobs,))


        sell_flag = jnp.tile(False, (num_lobs,))
        sell_num_shares = jnp.tile(-1, (num_lobs,))
        sell_price = jnp.tile(0.0, (num_lobs,))

        agent_state_content = {"cash": cash, "shares": shares, "beliefs": beliefs, "buy_flag": buy_flag, "buy_num_shares": buy_num_shares, 
                               "buy_price": buy_price, "sell_flag": sell_flag, "sell_num_shares": sell_num_shares, "sell_price": sell_price}
        agent_state = State(content=agent_state_content)
        agent_params = Params(content={"starting_price": starting_price, "belief_span": belief_span})

        return Trader(agent_type=type, params=agent_params, id=id, active_state=active_state, state=agent_state, policy=policy, key=key, age = 0.0)
    
    @staticmethod
    def step_agent(agent, input, step_params):
        # update cash, shares
        cash = agent.state.content["cash"]
        shares = agent.state.content["shares"]
        beliefs = agent.state.content["beliefs"]

        cash_diff = input.content["cash_diff"]
        shares_diff = input.content["shares_diff"]
        
        cash = jnp.maximum(cash + cash_diff, 0.0)  # Ensure cash and shares does not go negative
        shares = jnp.maximum(shares + shares_diff, 0)
        
        # for now just a noisy trader
        key, *dec_keys = jax.random.split(agent.key, 8)
        buy_flag = jax.random.uniform(dec_keys[0], (NUM_LOBS,), minval=0.0, maxval=1.0) < 0.5
        buy_num_shares = jax.random.randint(dec_keys[1], (NUM_LOBS,), minval=1, maxval=10)
        buy_price = beliefs - jax.random.uniform(dec_keys[2], (NUM_LOBS,), minval=-2.0, maxval=5.0)

        sell_flag = jax.random.uniform(dec_keys[3], (NUM_LOBS,), minval=0.0, maxval=1.0) < 0.5
        sell_num_shares = jax.random.randint(dec_keys[4], (NUM_LOBS,), minval=1, maxval=10)
        sell_price = beliefs + jax.random.uniform(dec_keys[5], (NUM_LOBS,), minval=-2.0, maxval=5.0)

        beliefs = beliefs + jax.random.uniform(dec_keys[6], (NUM_LOBS,), minval=-10.0, maxval=10.0)  # small noise to beliefs

        # update agent state
        agent_state_content = {"cash": cash, "shares": shares, "beliefs": beliefs, "buy_flag": buy_flag, 
                               "buy_num_shares": buy_num_shares, "buy_price": buy_price, 
                               "sell_flag": sell_flag, "sell_num_shares": sell_num_shares, "sell_price": sell_price}
        agent_state = State(content=agent_state_content)
        return agent.replace(state=agent_state, key=key, age=agent.age + step_params.content["dt"])
    

@struct.dataclass
class LOB:
    buy_LOB: Set
    sell_LOB: Set
    price: jnp.ndarray # a history of prices, shape (num_time_steps, 1)



