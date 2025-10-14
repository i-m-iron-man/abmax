import jax
import jax.numpy as jnp
import jax.numpy as jnp
import jax.random as random
from flax import struct

import sys
sys.path.append('../../')

from structs import *
from functions import *

class Animal:
    pass
class Grass:
    pass



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