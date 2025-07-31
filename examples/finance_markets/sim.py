import sys

sys.path.append('/Users/siddarth.chaturvedi/Desktop/source/abmax_git/abmax')


from structs import *
from functions import *
import jax.numpy as jnp
import jax.random as random
import jax
from flax import struct

from contexts.finance.structs import Order, Trader, LOB
from contexts.finance.functions import jit_match_orders, jit_get_order_add_params
import contexts.finance.structs as contexts_finance_structs

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
palette = "viridis"
sns.set_palette(palette)

DT = 1.0 # Time step for simulation
KEY = jax.random.PRNGKey(1)
NUM_STEPS = 100

# Order params
ORDER_AGENT_TYPE = 0
MAX_NUM_ORDERS = 1000
BUY_ORDER_POLARITY = jnp.array([0], dtype=jnp.int32)
SELL_ORDER_POLARITY = jnp.array([1], dtype=jnp.int32)
STARTING_PRICE = 100.0
MIN_ORDER_PRICE = 0.0
MAX_ORDER_PRICE = 1000.0
NUM_LOBS = 5
contexts_finance_structs.NUM_LOBS = NUM_LOBS  # Ensure this is set for the LOB creation

#trader params
MAX_NUM_TRADERS = 10
TRADER_AGENT_TYPE = 1

Market_create_params = Params(content={"num_traders": MAX_NUM_TRADERS, "trader_agent_type": TRADER_AGENT_TYPE, "starting_price": STARTING_PRICE, "max_num_orders": MAX_NUM_ORDERS,
                                       "order_agent_type": ORDER_AGENT_TYPE, "buy_order_polarity": BUY_ORDER_POLARITY, "sell_order_polarity": SELL_ORDER_POLARITY, "min_order_price": MIN_ORDER_PRICE, 
                                       "max_order_price": MAX_ORDER_PRICE})

@struct.dataclass
class Market:
    LOB: LOB
    traders: Set
    
    @staticmethod
    def create_market(params, key):
        key, *subkeys = random.split(key, 3)
        
        # step 1 creating traders
        num_traders = params.content["num_traders"]
        
        starting_price = params.content["starting_price"]
        starting_price_arr = jnp.tile(starting_price, (num_traders, 1))  # shape (num_traders, 1)
        belief_span_arr = jax.random.uniform(subkeys[0], (num_traders, 1), minval=0.1, maxval=0.5).reshape(-1)  # random belief span between 0.1 and 0.5
        

        trader_create_params = Params(content={"starting_price": starting_price_arr, "belief_span": belief_span_arr})

        traders = create_agents(Trader, trader_create_params, num_traders, num_traders, params.content["trader_agent_type"], subkeys[1])
        trader_set = Set(num_agents=num_traders, num_active_agents=num_traders, agents=traders, id=0, set_type=2,
                 params=None, state=None, policy=None, key=None)
        
        #step 2 create buy and sell LOBs
        num_lobs = NUM_LOBS
        key, *lob_keys = random.split(subkeys[1], num_lobs + 1)
        lob_keys = jnp.array(lob_keys)    
        
        def create_lobs(params, key):
            starting_price = params.content["starting_price"]
            max_num_orders = params.content["max_num_orders"]
            
            min_order_price = params.content["min_order_price"]
            max_order_price = params.content["max_order_price"]
            min_order_price_arr = jnp.tile(min_order_price, (max_num_orders, ))
            max_order_price_arr = jnp.tile(max_order_price, (max_num_orders, ))
            
            buy_polarity_arr = jnp.tile(params.content["buy_order_polarity"], (max_num_orders, 1))

            buy_create_params = Params(content={"polarity": buy_polarity_arr, "min_order_price": min_order_price_arr, "max_order_price": max_order_price_arr})
            num_active_buy_orders = 0

            # create buy orders
            buy_orders = create_agents(Order, buy_create_params, max_num_orders, num_active_buy_orders, params.content["order_agent_type"], key)

            buy_LOB = Set(num_agents=max_num_orders, num_active_agents=0, agents=buy_orders, id=0, set_type=0, params=None, state=None, policy=None, key=None)
            
            #create sell orders
            sell_polarity_arr = jnp.tile(params.content["sell_order_polarity"], (max_num_orders, 1))
            sell_create_params = Params(content={"polarity": sell_polarity_arr, "min_order_price": min_order_price_arr, "max_order_price": max_order_price_arr})
            
            num_active_sell_orders = 0
            sell_orders = create_agents(Order, sell_create_params, max_num_orders, num_active_sell_orders, params.content["order_agent_type"], key)
            sell_LOB = Set(num_agents=max_num_orders, num_active_agents=0, agents=sell_orders, id=0, set_type=1, params=None, state=None, policy=None, key=None)

            return LOB(buy_LOB=buy_LOB, sell_LOB=sell_LOB, price= jnp.array([starting_price]))
        lobs = jax.vmap(create_lobs, in_axes=(None, 0))(params, lob_keys)
        return Market(LOB=lobs, traders=trader_set)
    
    @staticmethod
    def get_empty_order_mask(orders:Agent):
        return orders.state.content["remove_flag"].reshape(-1)
    
    @staticmethod
    def get_lob_price(buy_lob, sell_lob):
        return jnp.mean(jnp.array([buy_lob.agents.params.content["price"][0], sell_lob.agents.params.content["price"][0]]), axis=0)
        #min_sell_price = jnp.min(sell_lob.agents.params.content["price"])
        #max_buy_price = jnp.max(buy_lob.agents.params.content["price"])
        #return min_sell_price + max_buy_price / 2.0  # average of the best buy and sell prices

    @staticmethod
    @jax.jit
    def step_market(market, _t):
        # step 1: match orders in lobs
        buy_orders, sell_orders, buy_order_step_input, sell_order_step_input, traders_cash_change, traders_shares_change  = jax.vmap(jit_match_orders,in_axes=(0,0,None))(market.LOB.buy_LOB.agents, market.LOB.sell_LOB.agents, market.traders.agents)
        buy_lobs = market.LOB.buy_LOB.replace(agents=buy_orders)
        sell_lobs = market.LOB.sell_LOB.replace(agents=sell_orders)
        
        lob_prices = jax.vmap(Market.get_lob_price)(buy_lobs, sell_lobs)

        # step 2: step the trader agents
        traders_cash_change = jnp.sum(traders_cash_change, axis=0).reshape(-1, 1)
        traders_shares_change = jnp.transpose(traders_shares_change)  # shape (num_traders, num_lobs)
        
        traders_step_input = Signal(content={"cash_diff": traders_cash_change,"shares_diff": traders_shares_change})
        step_params = Params(content={"dt": DT})
        trader_set = jit_step_agents(Trader.step_agent, input=traders_step_input, step_params=step_params, set=market.traders)

        #step 3 step orders
        buy_lobs = jax.vmap(jit_step_agents, in_axes=(None, None, 0, 0))(Order.step_agent, step_params, buy_order_step_input, buy_lobs)
        sell_lobs = jax.vmap(jit_step_agents, in_axes=(None, None, 0, 0))(Order.step_agent, step_params, sell_order_step_input, sell_lobs)

        # step 5: add new orders to the LOBs 
        buy_add_params, sell_add_params, num_buy_orders, num_sell_orders = jit_get_order_add_params(trader_set, buy_lobs, sell_lobs)
        #buy_lobs = jax.vmap(jit_add_agents, in_axes=(None, 0, 0, 0))(Order.add_agent, buy_add_params, num_buy_orders, buy_lobs)
        #sell_lobs = jax.vmap(jit_add_agents, in_axes=(None, 0, 0, 0))(Order.add_agent, sell_add_params, num_sell_orders, sell_lobs)
        buy_lobs = jax.vmap(jit_set_agents_sci, in_axes=(None, 0, 0, 0))(Order.add_agent_sci, buy_add_params, num_buy_orders, buy_lobs)
        sell_lobs = jax.vmap(jit_set_agents_sci, in_axes=(None, 0, 0, 0))(Order.add_agent_sci, sell_add_params, num_sell_orders, sell_lobs)
        
        # step 4: remove orders that are ready to be removed
        buy_orders_remove_mask = jax.vmap(Market.get_empty_order_mask)(buy_lobs.agents)
        buy_remove_mask_params = Params(content={"set_mask": buy_orders_remove_mask})
        sell_orders_remove_mask = jax.vmap(Market.get_empty_order_mask)(sell_lobs.agents)
        sell_remove_mask_params = Params(content={"set_mask": sell_orders_remove_mask})
        remove_params = Params(content={"min_order_price": MIN_ORDER_PRICE, "max_order_price": MAX_ORDER_PRICE})
        buy_lobs = jax.vmap(jit_set_agents_mask, in_axes=(None, None, 0, None, 0))(Order.remove_agent, remove_params, buy_remove_mask_params, -1, buy_lobs)
        sell_lobs = jax.vmap(jit_set_agents_mask, in_axes=(None, None, 0, None, 0))(Order.remove_agent, remove_params, sell_remove_mask_params, -1, sell_lobs)


        # step 6: update the market state
        LOB = market.LOB.replace(buy_LOB=buy_lobs, sell_LOB=sell_lobs, price=lob_prices)
        return market.replace(LOB=LOB, traders=trader_set), (lob_prices, buy_lobs.num_active_agents, sell_lobs.num_active_agents)

def run_scan(market, num_steps):
    
    ts = jnp.arange(num_steps)
    def scan_step(market, t):
        return Market.step_market(market, t)
    market, plot_data = jax.lax.scan(scan_step, market, ts)
    return market, plot_data

if __name__ == "__main__":
    # Create the market
    market = Market.create_market(Market_create_params, KEY)
    # Run the market simulation for a number of steps
    market, plot_data = run_scan(market, NUM_STEPS)
    price_history, num_buy_orders, num_sell_orders = plot_data
    print("Final Price History:", price_history.shape)
    price_history = price_history.reshape(-1, NUM_LOBS)  # Transpose to shape (num_steps, num_lobs)
    print("Final Price History:", price_history)
    # Plot the price history for each LOB deleting the first entry which is the initial price
    price_history = price_history[1:, :]  # Remove the initial price entry
    print("Price History Shape:", price_history.shape)
    
    # plot the price history
    fig, ax = plt.subplots(figsize=(15,10))
    for i in range(NUM_LOBS):
        ax.plot(price_history[:, i], label=f"LOB {i}")
    ax.set_xlabel("time step", fontsize=50)
    ax.set_ylabel("market price", fontsize=50)
    ax.legend(fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig('./LOB_prices.svg', bbox_inches='tight')
    plt.show()

    fig, ax = plt.subplots(figsize=(15,10))
    for i in range(NUM_LOBS):
        ax.plot(num_buy_orders[:, i], label=f"Buy LOB {i}")
    ax.set_xlabel("time step", fontsize=50)
    ax.set_ylabel("buy active orders", fontsize=50)
    ax.legend(fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig('./LOB_buy_orders.svg', bbox_inches='tight')
    plt.show()
    fig, ax = plt.subplots(figsize=(15,10))
    for i in range(NUM_LOBS):
        ax.plot(num_sell_orders[:, i], label=f"Sell LOB {i}")
    ax.set_xlabel("time step", fontsize=50)
    ax.set_ylabel("sell active orders", fontsize=50)
    ax.legend(fontsize=25)
    ax.tick_params(axis='both', which='major', labelsize=25)
    plt.savefig('./LOB_sell_orders.svg', bbox_inches='tight')
    plt.show()