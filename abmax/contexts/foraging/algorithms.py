import jax
import jax.numpy as jnp
from flax import struct

from abmax.structs import *
from abmax.functions import *
from ray_sensing import *

@struct.dataclass
class Patch:
    pass

@struct.dataclass
class Forager:
    pass

def agent_interactions(foragers:Forager, patches:Patch):

    def forager_patches_interaction(forager, patches):
        xs_patches = patches.state.content['x']  # an array of shape (num_patches,)
        ys_patches = patches.state.content['y']
        x_forager = forager.state.content['x']  # a 1x1 array
        y_forager = forager.state.content['y']

        dists = jnp.linalg.norm(jnp.stack((xs_patches - x_forager, ys_patches - y_forager), axis=1), axis=1).reshape(-1)  # distances from the forager to the patches
        is_near = jnp.where(dists < patches.params.content['radius'], 1.0, 0.0)  # 1 if the forager is near the patch, 0 otherwise
        
        #energy_offers = jnp.multiply(is_near, patches.state.content['energy_offer'].reshape(-1))  # energy offers from the patches
        
        is_in_patch = jnp.sum(is_near)  # how many patches the forager is in
        
        return is_in_patch, is_near#energy_offers
    
    def forager_forager_interaction(forager, foragers):
        xs_foragers = foragers.state.content['x']  # an array of shape (num_foragers,)
        ys_foragers = foragers.state.content['y']
        ids_foragers = foragers.id  # an array of shape (num_foragers,)

        x_forager = forager.state.content['x']  # a 1x1 array
        y_forager = forager.state.content['y']
        id_forager = forager.id  # a 1x1 array
        
        dist = jnp.linalg.norm(jnp.stack((xs_foragers - x_forager, ys_foragers - y_forager), axis=1), axis=1).reshape(-1)
        cond = jnp.logical_and(dist < forager.params.content['radius'], ids_foragers != id_forager)  # True if the forager is near another forager and not itself
        is_near = jnp.where(cond, 1.0, 0.0)  # 1 if the forager is near another forager, 0 otherwise
        is_in_forager = jnp.sum(is_near)  # how many other foragers the forager is in
        return is_in_forager
    
    is_in_patches, is_near_matrix = jax.vmap(forager_patches_interaction, in_axes=(0, None))(foragers, patches)
    is_in_foragers = jax.vmap(forager_forager_interaction, in_axes=(0, None))(foragers, foragers)

    is_in_sum = is_in_patches + is_in_foragers

    #energy_in_foragers = jnp.sum(energy_offers, axis=1).reshape(-1)  # sum the energy offers from the patches for each forager
    is_energy_out_patches = jnp.any(is_near_matrix, axis=0)  # t/f if a patch is being foraged by any forager

    num_foragers_present_per_patch = jnp.maximum(jnp.sum(is_near_matrix, axis=0),1.0)  # number of foragers present at each patch
    # point of maximum with 1.0: to avoid division by zero, and if there is 1 forager, it will just be 1.0
    energy_sharing_matrix = jnp.divide(is_near_matrix, num_foragers_present_per_patch)  # each forager gets an equal share of the patch's energy offer
    energy_in_foragers = jnp.multiply(energy_sharing_matrix, patches.state.content['energy_offer'].reshape(-1))  # energy received by each forager from each patch
    energy_in_foragers = jnp.sum(energy_in_foragers, axis=1).reshape(-1)  # sum the energy offers from the patches for each forager
    return is_in_sum, is_energy_out_patches, energy_in_foragers#energy_in_foragers, energy_out_patches

jit_agent_interactions = jax.jit(agent_interactions)


def get_sensor_data(foragers:Forager, patches:Patch):
    agent_xs = jnp.concatenate((foragers.state.content['x'].reshape(-1), patches.state.content['x'].reshape(-1)))
    agent_ys = jnp.concatenate((foragers.state.content['y'].reshape(-1), patches.state.content['y'].reshape(-1)))
    agent_rads = jnp.concatenate((foragers.params.content['radius'].reshape(-1), patches.params.content['radius'].reshape(-1)))
    agent_energies = jnp.concatenate((foragers.state.content['energy'].reshape(-1), patches.state.content['energy'].reshape(-1)))
    agent_types = jnp.concatenate((foragers.agent_type, patches.agent_type))

    points = jax.vmap(Point)(agent_xs, agent_ys)
    circles = jax.vmap(Circle)(points, agent_rads)

    type_sensor_values = jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # empty, forager, patch, used with lax.select

    def for_each_forager(forager):
        forager_pos = (forager.state.content['x'][0], forager.state.content['y'][0], forager.state.content['ang'][0])
        rays = generate_rays(forager_pos, RAY_SPAN, RAY_MAX_LENGTH)

        def for_each_ray(ray):
            intercepts =  jax.vmap(jit_get_ray_circle_collision, in_axes=(None, 0))(ray, circles)
            min_dist_indx = jnp.argmin(intercepts)
            
            min_dist = intercepts[min_dist_indx]
            
            sensed_energy, sensed_type = jax.lax.cond(min_dist<ray.length, 
                                                      lambda _: (agent_energies[min_dist_indx], agent_types[min_dist_indx]),
                                                      lambda _: (0.0, 0),  # 0 for empty, 1 for forager, 2 for patch
                                                      None)
            
            sensed_type_value = type_sensor_values[sensed_type]  # get the type sensor value based on the sensed type
            sensed_energy_channels = sensed_energy*sensed_type_value  # energy is only sensed if the type is forager or patch

            return jnp.concatenate((jnp.array([min_dist]), sensed_energy_channels))
        
        return jax.vmap(for_each_ray)(rays).reshape(-1)
    return jax.vmap(for_each_forager)(foragers)

jit_get_sensor_data = jax.jit(get_sensor_data)
