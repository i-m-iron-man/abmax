import jax
import jax.numpy as jnp
from flax import struct

from abmax.structs import *
from abmax.functions import *

@struct.dataclass
class Order:
    pass

@struct.dataclass
class Trader:
    pass

def match_orders(buy_orders: Order, sell_orders: Order, traders: Trader):
    """
    Match buy and sell orders in the order book.
    This function matches buy and sell orders based on their prices and updates the traders' cash and shares accordingly.
    After sorting of buy and sell orders respectively, the matching is vectorized over the buy orders.
    Args:
        buy_orders (Order): Buy orders in the order book.
        sell_orders (Order): Sell orders in the order book.
        traders (Trader): Traders who are placing the orders.
    Returns:
        Tuple[Order, Order, Signal, Signal, jnp.ndarray, jnp.ndarray]:
            - buy_orders_sorted (Order): Sorted buy orders after matching.
            - sell_orders_sorted (Order): Sorted sell orders after matching.
            - buy_order_step_input (Signal): Signal containing the number of shares removed from each buy order.
            - sell_order_step_input (Signal): Signal containing the number of shares removed from each sell order.
            - traders_cash_change (jnp.ndarray): Change in cash for each trader.
            - traders_shares_change (jnp.ndarray): Change in shares for each trader.
    """
    # step 1 sort sell orders according to increasing price and buy orders according to decreasing price, first element is the best price
    buy_orders_sorted, b_indx = jit_sort_agents(quantity=-1*buy_orders.params.content["price"], agents=buy_orders)
    sell_orders_sorted, s_indx = jit_sort_agents(quantity=sell_orders.params.content["price"], agents=sell_orders)

    #step 2 compute the cumulative number of shares
    buy_cumulative_shares = jnp.cumsum(buy_orders_sorted.state.content["num_shares"]).reshape(-1)
    sell_cumulative_shares = jnp.cumsum(sell_orders_sorted.state.content["num_shares"]).reshape(-1)

    # step 3 calculate the transaction for each buy order
    def for_each_buy_order(buy_order, buy_cumulative_share, sell_orders, sell_cumulative_shares):
        
        #step 3.1 get max transaction price for the buy order and anything above that price is not reachable by the buy order
        transaction_mask = jnp.where(buy_order.params.content["price"] < sell_orders.params.content["price"],0,1).reshape(-1) #0-> not reachable, 1-> reachable
        
        # step 3.2 multiply the cumulative shares by the transaction mask, this will set the cumulative shares to 0 for all the unreachable orders
        sell_cumulative_shares = jnp.multiply(sell_cumulative_shares, transaction_mask) # this will set the cumulative shares to 0 for all the unreachable orders
            
        num_sell_order_shares = sell_orders.state.content["num_shares"].reshape(-1)
        num_sell_order_shares = jnp.multiply(num_sell_order_shares, transaction_mask)

        # step 3.3 now remove the share from the sell orders that will be matched by buy orders with more priority than the current buy order
        # for this we need to know how many shares have more priority than the current buy order, that = buy_cumulative_share - buy_order.state.content["num_shares"]
        num_higher_priority_shares = buy_cumulative_share - buy_order.state.content["num_shares"][0]
        
        new_sell_cumulative_shares = jnp.maximum(sell_cumulative_shares - num_higher_priority_shares, 0)
        new_num_sell_order_shares = jnp.minimum(num_sell_order_shares, new_sell_cumulative_shares)

        # step 3.5 now using the same method as above remove the shares of this buy order
        new_sell_cumulative_shares = jnp.maximum(new_sell_cumulative_shares - buy_order.state.content["num_shares"][0], 0)
        new_new_num_sell_order_shares = jnp.minimum(new_num_sell_order_shares, new_sell_cumulative_shares)

        share_change = new_num_sell_order_shares - new_new_num_sell_order_shares
        cash_change = jnp.multiply(share_change, sell_orders.params.content["price"].reshape(-1))

        return share_change, cash_change

    share_change, cash_change = jax.vmap(for_each_buy_order, in_axes=(0, 0, None, None))(buy_orders_sorted, buy_cumulative_shares, sell_orders_sorted, sell_cumulative_shares)

    # use change to caculate change in shares and cash for traders
    # change in buy and sell shares, each row is for a buy order and each column is for a sell order
    
    def for_each_trader(trader):
        buy_orders_mask = jnp.where(trader.id == buy_orders_sorted.params.content["trader_id"].reshape(-1), 1, 0)
        sell_orders_mask = jnp.where(trader.id == sell_orders_sorted.params.content["trader_id"].reshape(-1), 1, 0)

        # when buying, first mask then matrix -> mask x matrix multiplication
        cash_change_buy = jnp.sum(jnp.matmul(buy_orders_mask, cash_change))
        shares_change_buy = jnp.sum(jnp.matmul(buy_orders_mask, share_change))

        # when selling, first matrix then mask -> matrix x mask multiplication
        cash_change_sell = jnp.sum(jnp.matmul(cash_change, sell_orders_mask))
        shares_change_sell = jnp.sum(jnp.matmul(share_change, sell_orders_mask))

        # update the trader cash and shares change, when selling add cash and remove shares, when buying remove cash and add shares
        trader_cash_change = cash_change_sell - cash_change_buy
        trader_shares_change = shares_change_buy - shares_change_sell
        return trader_cash_change, trader_shares_change
    
    traders_cash_change, traders_shares_change = jax.vmap(for_each_trader)(traders)

    #traders_step_input = Signal(content={"cash_diff": traders_cash_change.reshape(-1, 1),"shares_diff": traders_shares_change.reshape(-1, 1)})
    
    buy_shares_removed = share_change.sum(axis=1).reshape(-1, 1)  # total shares removed for each buy order
    sell_shares_removed = share_change.sum(axis=0).reshape(-1, 1)  # total shares removed for each sell order

    buy_order_step_input = Signal(content={"num_shares_remove": buy_shares_removed})
    sell_order_step_input = Signal(content={"num_shares_remove": sell_shares_removed})

    
    return buy_orders_sorted, sell_orders_sorted, buy_order_step_input, sell_order_step_input, traders_cash_change, traders_shares_change


jit_match_orders = jax.jit(match_orders)

def select_traders(trading_flags):
    trading_flags = trading_flags.reshape(-1)
    selected_mask = jnp.where(trading_flags , 1, 0)
    sort_selected_indx = jnp.argsort(-1*selected_mask)
    num_selected = jnp.sum(selected_mask)
    return num_selected, sort_selected_indx
jit_select_traders = jax.jit(select_traders)

def get_price_list(price_list_arr, indx):
    '''
    Get the price list for the given indices
    '''
    return jnp.take(price_list_arr, indx).reshape(-1, 1)

def get_num_shares_list(num_shares_arr, indx):
    '''
    Get the number of shares list for the given indices
    '''
    return jnp.take(num_shares_arr, indx).reshape(-1, 1)

def get_empty_orders(orders:Agent, select_params:Params):
    return jnp.logical_not(orders.active_state)  # Selects the slots where orders are not active


def get_order_add_params_SCI(trader_set, buy_LOB, sell_LOB):
    num_buy_orders, buy_trader_id = jax.vmap(jit_select_traders, in_axes=(1))(trader_set.agents.state.content["buy_flag"])

    buy_price_list = jax.vmap(get_price_list, in_axes=(1,0))(trader_set.agents.state.content["buy_price"], buy_trader_id)
    buy_num_shares_list = jax.vmap(get_num_shares_list, in_axes=(1,0))(trader_set.agents.state.content["buy_num_shares"], buy_trader_id)

    num_sell_orders, sell_trader_id = jax.vmap(jit_select_traders, in_axes=(1))(trader_set.agents.state.content["sell_flag"])
    
    sell_price_list = jax.vmap(get_price_list, in_axes=(1,0))(trader_set.agents.state.content["sell_price"], sell_trader_id)
    sell_num_shares_list = jax.vmap(get_num_shares_list, in_axes=(1,0))(trader_set.agents.state.content["sell_num_shares"], sell_trader_id)

    num_empty_buy_orders, inactive_buy_order_index = jax.vmap(jit_select_agents, in_axes=(None, None, 0))(get_empty_orders, None, buy_LOB)
    num_empty_sell_orders, inactive_sell_order_index = jax.vmap(jit_select_agents, in_axes=(None, None, 0))(get_empty_orders, None, sell_LOB)

    num_buy_orders = jnp.minimum(num_buy_orders, num_empty_buy_orders)
    num_sell_orders = jnp.minimum(num_sell_orders, num_empty_sell_orders)

    buy_add_params = Params(content={"set_indx": inactive_buy_order_index,
                                     "price_list": buy_price_list,
                                     "num_shares_list": buy_num_shares_list, 
                                     "trader_id_list": buy_trader_id})

    sell_add_params = Params(content={"set_indx": inactive_sell_order_index,
                                      "price_list": sell_price_list,
                                      "num_shares_list": sell_num_shares_list,
                                      "trader_id_list": sell_trader_id})
    
    return buy_add_params, sell_add_params, num_buy_orders, num_sell_orders
jit_get_order_add_params = jax.jit(get_order_add_params_SCI)