import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from abmax.structs import Params
from sim_params import *
import numpy as np


import seaborn as sns
sns.set_theme(style="darkgrid")
palette = "viridis"
sns.set_palette(palette)



def render_trajectory(render_data, filename):
    boid_xs = render_data.content['boid_xs'] # shape (EP_LEN, NUM_AGENTS)
    boid_ys = render_data.content['boid_ys']
    boid_angs = render_data.content['boid_angs']
    #boid_energies = render_data.content['boid_energies']
    boid_roles = render_data.content['boid_roles']


    # Initial positions (for static elements)
    boid_init_xs = boid_xs[0, :]
    boid_init_ys = boid_ys[0, :]
    boid_init_angs = boid_angs[0, :]
    #boid_init_energies = boid_energies[0, :]
    boid_init_roles = boid_roles[0, :]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Trajectory Visualization")
    ax.set_xlim(-1000, 1000)
    ax.set_ylim(-1000, 1000)
    ax.set_aspect('equal')
    ax.set_xticks(jnp.arange(-1000, 1001, 100))
    ax.set_yticks(jnp.arange(-1000, 1001, 100))
    
    #color based on initial role, red->inactive, pink->suboptimal, green->grazing, blue->exchange
    boid_colors = np.zeros((boid_init_roles.shape[0], 4))
    for i in range(boid_init_roles.shape[0]):
        role = boid_init_roles[i]
        if role == 0:
            boid_colors[i] = [0.13, 0.13, 0.13, 0.8]  # Inactive - dark gray
        elif role == 1:
            boid_colors[i] = [0.89, 0.22, 0.76, 0.8]  # Suboptimal - pink
        elif role == 2:
            boid_colors[i] = [0.17, 0.63, 0.17, 0.8]  # Grazing - green
        elif role == 3:
            boid_colors[i] = [0.12, 0.47, 0.71, 0.8]  # Exchange - blue

    # boids
    boid_scatter = ax.scatter(boid_init_xs, boid_init_ys, c=boid_colors, s=AGENT_RADIUS*4, alpha=0.8, linewidths=1.5)
    boid_quiver = ax.quiver(boid_init_xs, boid_init_ys, jnp.cos(boid_init_angs), jnp.sin(boid_init_angs), color=boid_colors, scale=100.0)

    def update(frame):
        #boid_colors = plt.cm.viridis(boid_energies[frame, :] / MAX_ENERGY)
        #boid_colors[:NUM_ACTIVE_AGENTS, :] = [1.0, 0.0, 0.0, 0.8]
        for i in range(boid_roles.shape[1]):
            role = boid_roles[frame, i]
            if role == 0:
                boid_colors[i] = [0.13, 0.13, 0.13, 0.8]  # Inactive - dark gray
            elif role == 1:
                boid_colors[i] = [0.89, 0.22, 0.76, 0.8]  # Suboptimal - pink
            elif role == 2:
                boid_colors[i] = [0.17, 0.63, 0.17, 0.8]  # Grazing - green
            elif role == 3:
                boid_colors[i] = [0.12, 0.47, 0.71, 0.8]  # Exchange - blue
        boid_scatter.set_offsets(jnp.vstack((boid_xs[frame,:], boid_ys[frame,:])).T)
        boid_scatter.set_facecolor(boid_colors)
        boid_quiver.set_offsets(jnp.vstack((boid_xs[frame,:], boid_ys[frame,:])).T)
        boid_quiver.set_UVC(jnp.cos(boid_angs[frame,:]), jnp.sin(boid_angs[frame,:]))
        boid_quiver.set_color(boid_colors)
        #bar container has no set_height method, need to get the rectangle from the container and set its height
        
        # Return all artists that need to be redrawn
        return boid_scatter, boid_quiver
    ani = FuncAnimation(fig, update, frames=range(boid_xs.shape[0]), blit=True)
    ani.save(filename+'.mp4', writer='ffmpeg', fps=60)


if __name__ == "__main__":
    mode = 'mean'  # 'mean' or 'best'
    seed = '7_final' # '7', '11', '5'
    boid_xs = jnp.load("./test_data/trajectories/seed_"+seed+"/"+mode+"/rendering_boids_xs.npy")
    boid_ys = jnp.load("./test_data/trajectories/seed_"+seed+"/"+mode+"/rendering_boids_ys.npy")
    boid_angs = jnp.load("./test_data/trajectories/seed_"+seed+"/"+mode+"/rendering_boids_angs.npy")
    #boid_energies = jnp.load("./test_data/trajectories/seed_"+seed+"/"+mode+"/rendering_boids_energies.npy")
    boid_roles = jnp.load("./test_data/trajectories/seed_"+seed+"/"+mode+"/rendering_boid_roles.npy")

    boid_xs = jnp.reshape(boid_xs, (boid_xs.shape[0], boid_xs.shape[1], boid_xs.shape[2]))
    boid_ys = jnp.reshape(boid_ys, (boid_ys.shape[0], boid_ys.shape[1], boid_ys.shape[2]))
    boid_angs = jnp.reshape(boid_angs, (boid_angs.shape[0], boid_angs.shape[1], boid_angs.shape[2]))
    #boid_energies = jnp.reshape(boid_energies, (boid_energies.shape[0], boid_energies.shape[1], boid_energies.shape[2]))
    boid_roles = jnp.reshape(boid_roles, (boid_roles.shape[0], boid_roles.shape[1], boid_roles.shape[2]))

    
    num_trajectories = boid_xs.shape[0]
    trajectory_length = boid_xs.shape[1]
    print(f"Number of trajectories: {num_trajectories}, Trajectory length: {trajectory_length}")

    
    '''
    for traj_idx in range(num_trajectories):
        render_data = Params(
            content={
            'boid_xs': boid_xs[traj_idx, :, :],
            'boid_ys': boid_ys[traj_idx, :, :],
            'boid_angs': boid_angs[traj_idx, :, :],
            'boid_energies': boid_energies[traj_idx, :, :]
            })
     
        
        print(f"Rendering trajectory {traj_idx+1}/{num_trajectories}...")
        render_trajectory(render_data, VIDEO_PATH + f"{mode}/trajectory_{traj_idx}")
    '''
    render_data = Params(
            content={
            'boid_xs': boid_xs[-1, :, :],
            'boid_ys': boid_ys[-1, :, :],
            'boid_angs': boid_angs[-1, :, :],
            'boid_roles': boid_roles[-1, :, :]
            #'boid_energies': boid_energies[-1, :, :]
            })
    
    render_trajectory(render_data, VIDEO_PATH + f"{mode}/trajectory_final_paper")
    

