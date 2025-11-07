# Foragax
<div align="center">
    <img src="https://github.com/i-m-iron-man/abmax/blob/master/examples/foragax/media/large_forage.gif" width="500"/>
</div>

Foragax is an extenstion of ABMax into the domain of many-agent foraging problems by considering foragers as circular active particles. It implements 3 types of entities:

- Foragers: Circular active particles that move around the environment to collect resources.
- Resources: Circular passive particles that represent food items to be collected by foragers.
- Walls: lines that represent obstacles and boundaries in the environment.

Foragers use occlusion-aware raycasting to sense their surroundings, allowing them to detect nearby entities.
Following is a screenshot of the simulation environment:
<div align="center">
    <img src="https://github.com/i-m-iron-man/abmax/blob/master/examples/foragax/media/foragax.png" width="500"/>
</div>

# Example Project
An example project, where swarming that emerges during foraging is linked to the internal amount of resource in the foragers, can be found below:
- [Paper](https://arxiv.org/abs/2510.18886)
- [Code](https://github.com/i-m-iron-man/abmax/blob/master/examples/foragax/train.py)


