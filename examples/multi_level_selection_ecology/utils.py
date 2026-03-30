# be careful while taking MEAN, always use the number of active agents.

from abmax.functions import *
import jax.numpy as jnp
import jax.random as random
import jax
from flax import struct
from evosax import CMA_ES

from sim_params import *
from agent import *
from ray_sensing import *

def get_sensor_data(boids):
    agent_xs = boids.state.content['x'].reshape(-1)
    agent_ys = boids.state.content['y'].reshape(-1)
    agent_energies = boids.state.content['energy'].reshape(-1)

    points = jax.vmap(Point)(agent_xs, agent_ys)
    circles = jax.vmap(Circle, in_axes=(0,None))(points, AGENT_RADIUS)

    def for_each_boid(boid):
        boid_pos = (boid.state.content['x'][0], boid.state.content['y'][0], boid.state.content['ang'][0])
        rays = generate_rays(boid_pos, RAY_SPAN, RAY_LENGTH)

        def for_each_ray(ray):
            intercepts =  jax.vmap(jit_get_ray_circle_collision, in_axes=(None, 0))(ray, circles)
            min_dist_indx = jnp.argmin(intercepts)
            
            min_dist = intercepts[min_dist_indx]
            
            sensed_energy = jax.lax.cond(min_dist<ray.length, lambda _: agent_energies[min_dist_indx],
                                                      lambda _: 0.0, None)
            
            return jnp.array([min_dist, sensed_energy], dtype=jnp.float32)
        
        return jax.vmap(for_each_ray)(rays).reshape(-1)
    return jax.vmap(for_each_boid)(boids)

jit_get_sensor_data = jax.jit(get_sensor_data)

def agent_interactions(boids:Boid):
    def boid_boid_interaction(boid, boids):
        xs_boids = boids.state.content['x']  # an array of shape (num_boids,)
        ys_boids = boids.state.content['y']
        ids_boids = boids.id  # an array of shape (num_boids,)

        x_boid = boid.state.content['x']  # a 1x1 array
        y_boid = boid.state.content['y']
        id_boid = boid.id  # a 1x1 array
        
        # get is_in
        dist = jnp.linalg.norm(jnp.stack((xs_boids - x_boid, ys_boids - y_boid), axis=1), axis=1).reshape(-1)
        cond = jnp.logical_and(dist < 2 * AGENT_RADIUS, ids_boids != id_boid)  # True if the boid is near another boid and not itself
        is_near = jnp.where(cond, 1.0, 0.0)  # 1 if the boid is near another boid, 0 otherwise
        is_in_boid = jnp.sum(is_near)  # how many other boids the boid is in
        
        # get energy transfer
        boid_boids_energy_diff = (boids.state.content['energy'] - boid.state.content['energy']).reshape(-1)  # energy difference between the boid and other boids
        boids_boid_energy_transfer = SNATCH_COEFFICIENT * jnp.multiply(is_near, boid_boids_energy_diff)  # energy transfer from other boids to the boid
        boids_boid_energy_transfer = jnp.clip(boids_boid_energy_transfer, -MAX_ENERGY_SNATCH, MAX_ENERGY_SNATCH)  # limit the energy transfer to a maximum value from each boid
        boids_boid_energy_transfer = jnp.sum(boids_boid_energy_transfer)  # sum the energy transfer from all other boids to the boid
        
        
        return is_in_boid, boids_boid_energy_transfer
    
    is_in_boids, boids_boid_energy_transfer = jax.vmap(boid_boid_interaction, in_axes=(0, None))(boids, boids)
    
    
    return is_in_boids, boids_boid_energy_transfer

jit_agent_interactions = jax.jit(agent_interactions)

def get_avg_movement(boids):
    # average movement of only active agents
    active_boid_mask = jnp.where(boids.active_state, 1.0, 0.0)
    num_active_boids = jnp.maximum(1e-8, jnp.sum(active_boid_mask))
    movements = jnp.multiply(active_boid_mask, boids.state.content['movement'].reshape(-1))
    total_movement = jnp.sum(movements)
    avg_movement = total_movement / num_active_boids
    return avg_movement.reshape(1,)
