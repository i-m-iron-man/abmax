import jax
import jax.numpy as jnp
import jax.numpy as jnp
import jax.random as random
from flax import struct

from abmax.structs import *
from abmax.functions import *

@struct.dataclass
class Cell:
    pass

@struct.dataclass
class Car:
    pass

@struct.dataclass
class Road:
    pass

def XY_to_cell_id(X:jnp.array, Y:jnp.array, X_max:jnp.array, Y_max:jnp.array):

    X_cond = jnp.logical_and(X[0] < X_max[0], X[0] >= 0)
    Y_cond = jnp.logical_and(Y[0] < Y_max[0], Y[0] >= 0)

    cell_id = jax.lax.cond(jnp.logical_and(X_cond, Y_cond), 
                            lambda _: X + jnp.multiply(Y, X_max), 
                            lambda _: jnp.array([-1]),
                            None)
    return cell_id
jit_XY_to_cell_id = jax.jit(XY_to_cell_id)

def cell_id_to_XY(cell_id, X_max, Y_max):
    X_cond = jnp.logical_and(cell_id[0] < jnp.multiply(X_max, Y_max)[0], cell_id[0] >= 0)
    Y_cond = jnp.logical_and(cell_id[0] < jnp.multiply(X_max, Y_max)[0], cell_id[0] >= 0)
    XY = jax.lax.cond(jnp.logical_and(X_cond, Y_cond), 
                      lambda _: (jnp.mod(cell_id, X_max), jnp.floor_divide(cell_id, X_max)),
                      lambda _: (jnp.array([-1]), jnp.array([-1])),
                      None)
    return XY
jit_cell_id_to_XY = jax.jit(cell_id_to_XY)


def is_move(cars: Agent, cells: Agent):
    #check if the car is chosen by the cell
    def is_chosen(car: Agent, cells: Agent):
        requested_cell_id = car.state.content['request_cell_id']
        current_cell_id = car.state.content['current_cell_id']
        exit = cells.params.content['exit'][current_cell_id[0]][0]  # Check if the current cell is an exit cell
        chosen_car_id = cells.state.content['car_id'][requested_cell_id][0]
        

        chosen = jax.lax.cond(jnp.logical_and(requested_cell_id[0] >= 0, exit==0), 
                              lambda _: car.id == chosen_car_id[0], # Only to move if the cell chose the car id and the cell is not an exit cell
                              lambda _: False, # If no cell has been requested yet, car is not to move.
                              None)
        return chosen

    chosen_arr = jax.vmap(is_chosen, in_axes=(0, None))(cars, cells)
    return chosen_arr
jit_is_move = jax.jit(is_move)

def is_remove(cars: Agent, cells:Agent):
    # if the car is at the exit cell, and the light is green(1), then it can be removed.
    def for_a_car(car):
        current_cell_id = car.state.content['current_cell_id']
        cond = jnp.logical_and(cells.params.content['exit'][current_cell_id[0]][0] == 1, cells.state.content['light'][current_cell_id[0]][0] == 1)
        cond = jnp.logical_and(cond, car.active_state) # car is active
        remove_flag = jax.lax.cond(cond,
                                  lambda _: 1, # car can be removed
                                  lambda _: 0, # car cannot be removed
                              None)
        return remove_flag
    remove_flags = jax.vmap(for_a_car)(cars)
    return remove_flags
jit_is_remove = jax.jit(is_remove)


def select_empty_slots(cars:Agent, select_params:Params):
    return jnp.logical_not(cars.active_state)  # Selects the slots where cars are not active

def spawn_cars(road:Road, key: jax.random.PRNGKey):
    cells = road.cells.agents

    entry_cell_ids = jnp.array([0,1,2])
    key, *subkey = random.split(key,3)

    num_cars_add = jax.random.randint(subkey[0], (1,), minval=1, maxval=4)[0]  # Randomly choose how many cars to spawn
    

    shuffled_entry_cell_ids = jax.random.permutation(subkey[1], entry_cell_ids)
    num_cars_shuffled = jnp.take(cells.state.content['num_cars'], shuffled_entry_cell_ids)
    is_cell_empty = jnp.where(num_cars_shuffled == 0, 1, 0)

    sorted_car_indexes  = jnp.argsort(-1*is_cell_empty.reshape(-1))  # Sort the car indexes based on the number of cars in the entry cells
    sorted_shuffled_entry_cell_ids = jnp.take(shuffled_entry_cell_ids, sorted_car_indexes)  # Get the sorted entry cell ids 

    num_empty_cells = jnp.sum(is_cell_empty)  # Calculate the number of empty cells in the entry cells
    num_cars_add = jnp.minimum(num_cars_add, num_empty_cells)  # Ensure we don't add more cars than empty cells

    _, inactive_car_index = jit_select_agents(select_empty_slots, None, road.cars)

    #define cell set params using set_sci method
    cell_set_params = Params(content={'set_indx': sorted_shuffled_entry_cell_ids, 'car_id': inactive_car_index})

    #define car add params based on the set_sci method
    car_add_params = Params(content={'set_indx': inactive_car_index, 'current_cell_ids': sorted_shuffled_entry_cell_ids})

    return cell_set_params, car_add_params, num_cars_add, key

jit_spawn_cars = jax.jit(spawn_cars)