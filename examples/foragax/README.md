# Foragax
<div align="center">
    <img src="" width="250"/>
</div>

Foragax is an extenstion of ABMax into the domain of many-agent foraging problems by considering foragers as circular active particles. It implements 3 types of entities:

- Foragers: Circular active particles that move around the environment to collect resources.
- Resources: Circular passive particles that represent food items to be collected by foragers.
- Walls: lines that represent obstacles and boundaries in the environment.

Foragers use occlusion-aware raycasting to sense their surroundings, allowing them to detect nearby entities.