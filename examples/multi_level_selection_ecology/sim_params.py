import jax.numpy as jnp
import jax

# simulation update params
EP_LEN = 4000#20000#4000 #c
DT = 0.1

# distances
MAX_WORLD_X = 10000.0 # max travel distance in x direction
MAX_WORLD_Y = MAX_WORLD_X # max travel distance in y direction
MAX_SPAWN_X = 50.0#500.0
MAX_SPAWN_Y = 50.0#500.0
OFFSPRING_SPREAD= 30.0 # when an agent reproduces, the offspring will spawn within this radius of the parent
MAX_VEL = 20.0 # in either direction X or Y
MAX_ANG_VEL = jnp.pi/3 # max angular velocity

AGENT_RADIUS = 10.0

RAY_LENGTH = 300.0
RAY_RESOLUTION = 11 #19
RAY_SPAN = jnp.pi  # 360 degrees 5.0/sin(180/18 deg) = 28.7 units of no blind spot

# position update
NUM_ACTIONS = 2 # angular velocity, speed
DAMPING = 0.3 # you need active propulsion
NOISE_SCALE_ACC = 0.1 # position updates are noisy
ACTION_SCALE = 1.0
LINEAR_ACTION_SCALE = ACTION_SCALE * AGENT_RADIUS # 5.0
MOVEMENT_TIME_CONSTANT = 0.04 # for smoothing the movement of neighbors when calculating target movement

# energy update
MAX_ENERGY = 100.0 # max energy an agent can have
BAR_ENERGY_TIME_CONSTANT = 0.04 # will be used for smoothing the target energy over time 

GRAZING_COEFFICIENT = 0.1 # Constant rate of energy gain from grazing
MAX_GRAZING_ENERGY = 0.04 * MAX_ENERGY # max energy that can be gained from grazing at once
SNATCH_COEFFICIENT = 0.2 # portion of energy that can be snatched
MAX_ENERGY_SNATCH = 0.08 * MAX_ENERGY # max energy that can be snatched at once ->100*0.02 = 2.0


#Birth and death constants
REPRODUCTION_ENERGY_THRESHOLD = 0.2 * MAX_ENERGY # energy level at
REPRODUCION_TIME_THRESHOLD =  0.01 * EP_LEN # minimum time between reproductions
DEATH_ENERGY_THRESHOLD = 0.02 * MAX_ENERGY # energy level at which an agent dies
DEATH_TIME_THRESHOLD = 0.01 * EP_LEN # minimum time between deaths
OLD_AGE_ONSET = 400#5000#400
MAX_AGE = 600#6000#600

# fitness update
METABOLIC_COST_ANGULAR = 0.04
METABOLIC_COST_LINEAR = METABOLIC_COST_ANGULAR/ AGENT_RADIUS # making it proportional to radius
BASIC_METABOLIC_COST = 0.001 # idle cost 

# controller params
NUM_OBS = 2*RAY_RESOLUTION + 10 #  for  xdot, ydot, ang_dot, in_sum, average movement, movement, energy, grazing_energy, exchange_energy, metabolic_cost
NUM_NEURONS = 40 #50
NEURON_TIME_CONSTANT_SCALE = 10.0
BAR_NEURON_TIME_CONSTANT_SCALE = 0.04

MLP_SELECTOR_HIDDEN_SIZE = 16
NUM_INPUTS_SELECTOR = 8 # [bar_z_in, bar_z_out, synapse, bar_energy, bar_grazing_energy, bar_exchange_energy, bar_metabolic_cost, movement]
SELECTOR_NOISE_SCALE = 0.05
J_LIMIT = 3.0
MLP_SCALING = 0.1# to keep the MLP outputs in a reasonable range for modulating the J values

# training params
NUM_AGENTS = 50#100#50
NUM_ACTIVE_AGENTS = 5
NUM_WORLDS = 50
NUM_SCENARIOS = 2
NUM_GENERATIONS = 1000 # number of times both populations are trained against each other, in dual training mode. In single population training mode, this is the number of times the population is trained and evaluated.
ELITE_RATIO = 0.3
SIGMA_INIT = 0.1
POPULATION_SIZE = NUM_WORLDS
NUM_ES_PARAMS = NUM_NEURONS * (NUM_OBS + NUM_ACTIONS + 2) + MLP_SELECTOR_HIDDEN_SIZE * (NUM_INPUTS_SELECTOR + MLP_SELECTOR_HIDDEN_SIZE +3)
# CMAES for CTRNN: E:(NUM_NEURONS x NUM_OBS), D:(NUM_NEURONS x NUM_ACTIONS), B:(NUM_NEURONS,), tau: (NUM_NEURONS)
# Cmaes for selector: W1:(MLP_SELECTOR_HIDDEN_SIZE x NUM_INPUTS_SELECTOR), b1:(MLP_SELECTOR_HIDDEN_SIZE,), W2:(MLP_SELECTOR_HIDDEN_SIZE x MLP_SELECTOR_HIDDEN_SIZE), b2:(MLP_SELECTOR_HIDDEN_SIZE,), W3:(NUM_OUTPUTS_SELECTOR(1) x MLP_SELECTOR_HIDDEN_SIZE)

# useless 
SEEDS = [11,7,5,3,2,1] #useless
SEED_IDX = 1 #useless
KEY = jax.random.PRNGKey(SEEDS[SEED_IDX]) #useless
AGENT_TYPE = 1

#saving params
PARAM_PATH = f"./test_data/params/seed_{SEEDS[SEED_IDX]}/"
TRAJ_PATH = f"./test_data/trajectories/seed_{SEEDS[SEED_IDX]}/"
VIDEO_PATH = f"./test_data/videos/seed_{SEEDS[SEED_IDX]}/"




SIM_PARAMS_CONTENT = {"agent_params": {
    "agent_type": AGENT_TYPE,
    "num_agents": NUM_AGENTS,
    "num_active_agents": NUM_ACTIVE_AGENTS,
},
"policy_params": {
    "num_neurons": NUM_NEURONS,
    "num_obs": NUM_OBS,
    "num_actions": NUM_ACTIONS
}
}