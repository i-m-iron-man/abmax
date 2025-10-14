<div align="center">
    <img src="https://github.com/i-m-iron-man/abmax/blob/master/media/abmx_logo.png" width="250"/>
</div>
<div align="center">
    <img src="https://github.com/i-m-iron-man/abmax/blob/master/media/flocking.gif" width="250"/>
    <img src="https://github.com/i-m-iron-man/abmax/blob/master/media/sheep_wolf.gif" width="250"/>
    <img src="https://github.com/i-m-iron-man/abmax/blob/master/media/small_foragaing.gif" width="250"/>
</div>

Abmax is a general-purpose agent-based modeling(ABM) framework in Jax
It provides:
- Two algorithms to apply unique updates to a dynamic number of agents selected during run-time. Both are JIT friendly and can be vectorized across different number of models
    * Rank-Match (RM)
    * Sort-Count-Iterate (SCI)
- JAX-transformation-friendly data structures and functions that can be used to define sets of agents and their manipulations.
    * Selecting agents based on a run-time determined key.
    * Searching and sorting agents based on their attributes.
    * Updating an arbitrary number of agents to a specific state.
    * Stepping agents in a vectorized way.
    * Running multiple such simulations in parallel.
- Implementation of common algorithms used in ABM implemented in vmap and jit friendly way.

# Installation
```bash
pip install abmax
```
Dependencies:
- [Python >= 3.10](https://www.python.org/downloads/)
- [Jax >= 0.4.13](https://jax.readthedocs.io/en/latest/installation.html)
- [Flax >= 0.7.4](https://flax.readthedocs.io/en/latest/index.html)

# Benchmark
A comparison of the performance of Abmax with other ABM frameworks: [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/) and [Mesa](https://mesa.readthedocs.io/en/stable/) based on the [Wolf-Sheep (Grid space) and Bird-Flock (Continuous space) models](https://github.com/i-m-iron-man/ABMFrameworksComparison/tree/main). These simulations are run for 100 steps and the median time taken for 10 runs is logged in ms. The benchmark is run on a [gcn GPU node](https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+hardware)(Intel Xeon Platinum 8360Y + Nvidia A100) of the [Snelius cluster](https://www.surf.nl/en/services/snellius-the-national-supercomputer)
The number of initial agents for these simulations are as follows:
- Wolf-Sheep small: 600 sheep, 400 wolves on a 100 x 100 grid
- Wolf-Sheep large: 6000 sheep, 4000 wolves on a 1000 x 1000 grid

| Model | Agents.jl | Abmax RM | Abmax SCI | Mesa |
| ----- | ----- | ---- | ----- | ---- |
| Wolf-Sheep small | 14.93 | 50.26 | 726.78 | 1333.047
| Wolf-Sheep large | 685.03 | 3315.88 | 5455.01 | 170070.95

In Abmax, we can [run multiple simulations](https://github.com/i-m-iron-man/abmax/blob/master/benchmarks/wolf_sheep/benchmarks_vmap.py) in parallel because of automatic batching and vectorization. 
Here is a trend in running different numbers of wolf-sheep small models in parallel.

| Number of models | 10 | 20 | 50 | 100 | 200 | 500 |
| ----------------- | -- | -- | -- | --- | --- | --- |
| time taken (s) | 5.75 | 6.81 | 7.32 | 8.52 | 8.617 | 14.32 |

Note: All times that are reported, are excluding the model setup time.


# Tutorial
A basic tutorial on how to use Abmax is available [here](https://github.com/i-m-iron-man/abmax/blob/master/tutorials/getting_started.ipynb)[Outdated]


# Citation
If you use Abmax in your work, please consider citing it as follows:
```
@misc{chaturvedi2025abmax,
    title={Abmax: A JAX-based Agent-based Modeling Framework},
    author={Siddharth Chaturvedi and Ahmed El-Gazzar and Marcel van Gerven},
    year={2025},
    eprint={2508.16508},
    archivePrefix={arXiv},
    primaryClass={cs.MA}
}


