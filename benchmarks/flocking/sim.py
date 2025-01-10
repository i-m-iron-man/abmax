
from abmax.structs import *
from abmax.functions import *
import jax.numpy as jnp
import jax.random as random
import jax

@struct.dataclass
class Bird(Agent):
    @staticmethod
    def create_agent(type, params, id, active_state, key):
        key, *subkeys = random.split(key, 5)
        X_pos_max = params.content['X_pos_max']
        Y_pos_max = params.content['Y_pos_max']

        agent_params_content = {"speed": params.content['speed'],
                                "cohere_factor": params.content['cohere_factor'],
                                "seperation": params.content['seperation'],
                                "separate_factor": params.content['seperate_factor'],
                                "match_factor": params.content['match_factor'],
                                "visual_distance": params.content['visual_distance'],
                                "X_pos_max": X_pos_max,
                                "Y_pos_max": Y_pos_max}
        
        agent_params = Params(content=agent_params_content)
        
        X_pos = random.uniform(subkeys[0], (1,), minval=0, maxval=X_pos_max)
        Y_pos = random.uniform(subkeys[1], (1,), minval=0, maxval=Y_pos_max)
        X_vel = random.uniform(subkeys[2], (1,), minval=-1, maxval=1)
        Y_vel = random.uniform(subkeys[3], (1,), minval=-1, maxval=1)
        state_content = {"X_pos": X_pos, "Y_pos": Y_pos, "X_vel": X_vel, "Y_vel": Y_vel}
        state = State(state_content)

        return Bird(id=id, active_state=active_state, age=0.0, agent_type=type,
                    params=agent_params, state=state, policy=None, key=key)
    @staticmethod
    def select_near_neighbours(agents_pos, select_params):
        agent_xs, agent_ys = agents_pos

        agent_x = select_params.content['X_pos']
        agent_y = select_params.content['Y_pos']
        theshold = select_params.content['threshold']
        dist = jnp.sqrt((agent_xs - agent_x)**2 + (agent_ys - agent_y)**2)
        return dist <= theshold
    
    @staticmethod
    def step_agent(agent, input, step_params):
        agents = step_params.content['agents']
        agents_xs = agents.state.content['X_pos'].reshape(-1)
        agents_ys = agents.state.content['Y_pos'].reshape(-1)
        agents_pos = (agents_xs, agents_ys)
        agents_x_vels = agents.state.content['X_vel'].reshape(-1)
        agents_y_vels = agents.state.content['Y_vel'].reshape(-1)
        dt = step_params.content['dt']

        #obtain the position of agents in the visual range
        visual_select_params = Params(content={"X_pos": agent.state.content['X_pos'],
                                        "Y_pos": agent.state.content['Y_pos'],
                                        "threshold": agent.params.content['visual_distance']})
        
        visual_mask = jnp.where(jax.jit(Bird.select_near_neighbours)(agents_pos, visual_select_params), 1, 0)
        num_visual_neighbours = jnp.maximum(jnp.sum(visual_mask), 1)

        seperation_select_params = Params(content={"X_pos": agent.state.content['X_pos'],
                                        "Y_pos": agent.state.content['Y_pos'],
                                        "threshold": agent.params.content['seperation']})
        
        seperation_mask = jnp.where(jax.jit(Bird.select_near_neighbours)(agents_pos, seperation_select_params), 1, 0)
        num_seperation_neighbours = jnp.maximum(jnp.sum(seperation_mask), 1)

        #cohere
        cohere_x = jnp.multiply(jnp.sum(jnp.multiply(agents_xs - agent.state.content['X_pos'], visual_mask)) / num_visual_neighbours, agent.params.content['cohere_factor'])
        cohere_y = jnp.multiply(jnp.sum(jnp.multiply(agents_ys - agent.state.content['Y_pos'], visual_mask)) / num_visual_neighbours, agent.params.content['cohere_factor'])

        #seperate
        seperate_x = jnp.multiply(jnp.sum(jnp.multiply(agent.state.content['X_pos'] - agents_xs, seperation_mask)) / num_seperation_neighbours, agent.params.content['separate_factor'])
        seperate_y = jnp.multiply(jnp.sum(jnp.multiply(agent.state.content['Y_pos'] - agents_ys, seperation_mask)) / num_seperation_neighbours, agent.params.content['separate_factor'])

        #match
        match_x = jnp.multiply(jnp.sum(jnp.multiply(agents_x_vels, visual_mask)) / num_visual_neighbours, agent.params.content['match_factor'])
        match_y = jnp.multiply(jnp.sum(jnp.multiply(agents_y_vels, visual_mask)) / num_visual_neighbours, agent.params.content['match_factor'])

        #update velocity
        new_x_vel = (agent.state.content['X_vel'] + cohere_x + seperate_x + match_x)/2.0
        new_y_vel = (agent.state.content['Y_vel'] + cohere_y + seperate_y + match_y)/2.0
        norm = jnp.sqrt(new_x_vel**2 + new_y_vel**2)
        new_x_vel = new_x_vel / norm
        new_y_vel = new_y_vel / norm

        #update position
        new_x = agent.state.content['X_pos'] + new_x_vel * agent.params.content['speed'] * dt
        new_y = agent.state.content['Y_pos'] + new_y_vel * agent.params.content['speed'] * dt

        #bouncing off the walls
        new_x, new_x_vel = jax.lax.cond(new_x[0] > agent.params.content['X_pos_max'], lambda _: (jnp.array([agent.params.content['X_pos_max']]), -1*new_x_vel), lambda _: (new_x, new_x_vel), None)
        new_x, new_x_vel = jax.lax.cond(new_x[0] < 0, lambda _: (jnp.array([0.0]), -1*new_x_vel), lambda _: (new_x, new_x_vel), None)
        new_y, new_y_vel = jax.lax.cond(new_y[0] > agent.params.content['Y_pos_max'], lambda _: (jnp.array([agent.params.content['Y_pos_max']]), -1*new_y_vel), lambda _: (new_y, new_y_vel), None)
        new_y, new_y_vel = jax.lax.cond(new_y[0] < 0, lambda _: (jnp.array([0.0]), -1*new_y_vel), lambda _: (new_y, new_y_vel), None)

        #update state
        new_state_content = {"X_pos": new_x, "Y_pos": new_y, "X_vel": new_x_vel, "Y_vel": new_y_vel}
        new_state = State(new_state_content)
        new_age = agent.age + dt
        return agent.replace(state=new_state, age=new_age)




def run_loop(set, num_steps):
    def step(set,x):
        step_params = Params(content={"agents": set.agents, "dt": 1.0})
        new_set = jit_step_agents(Bird.step_agent, step_params, input=None, set=set)
        agent_pos = (new_set.agents.state.content['X_pos'].reshape(-1), new_set.agents.state.content['Y_pos'].reshape(-1))
        return new_set, agent_pos
    new_set, agent_pos = jax.lax.scan(f=jax.jit(step), init=set, xs=None, length=num_steps)
    return agent_pos


X_pos_max = 200.0 
Y_pos_max = 200.0
speed = 2.0
cohere_factor = 0.4
separate_factor = 0.25
match_factor = 0.02
visual_distance = 10.0
seperation = 4.0
num_agents = 1000
num_active_agents = 1000
agent_type = 1
key = random.PRNGKey(0)
num_steps = 100

params_content_small = {"X_pos_max": jnp.tile(X_pos_max, num_agents),
                    "Y_pos_max": jnp.tile(Y_pos_max, num_agents),
                    "speed": jnp.tile(speed, num_agents),
                    "cohere_factor": jnp.tile(cohere_factor, num_agents),
                    "seperate_factor": jnp.tile(separate_factor, num_agents),
                    "match_factor": jnp.tile(match_factor, num_agents),
                    "visual_distance": jnp.tile(visual_distance, num_agents),
                    "seperation": jnp.tile(seperation, num_agents)}
params_small = Params(content=params_content_small)

bird_agents_small = create_agents(Bird, params_small, num_agents, num_active_agents, agent_type, key)
bird_set_small = Set(agents=bird_agents_small, num_agents=num_agents, num_active_agents=num_active_agents,
               id=0, set_type=0, params=None, state=None, policy=None, key=None)

X_pos_max = 500.0 
Y_pos_max = 500.0
num_agents = 10000
num_active_agents = 10000
visual_distance = 10.0

params_content_large = {"X_pos_max": jnp.tile(X_pos_max, num_agents),
                    "Y_pos_max": jnp.tile(Y_pos_max, num_agents),
                    "speed": jnp.tile(speed, num_agents),
                    "cohere_factor": jnp.tile(cohere_factor, num_agents),
                    "seperate_factor": jnp.tile(separate_factor, num_agents),
                    "match_factor": jnp.tile(match_factor, num_agents),
                    "visual_distance": jnp.tile(visual_distance, num_agents),
                    "seperation": jnp.tile(seperation, num_agents)}
params_large = Params(content=params_content_large)

bird_agents_large = create_agents(Bird, params_large, num_agents, num_active_agents, agent_type, key)
bird_set_large = Set(agents=bird_agents_large, num_agents=num_agents, num_active_agents=num_active_agents,
               id=0, set_type=0, params=None, state=None, policy=None, key=None)


def main(bird_set, num_steps):
    xs,ys = run_loop(bird_set, num_steps)
    return xs,ys

if __name__ == "__main__":
    xs,ys = main(bird_set_large, num_steps)
    print(xs,ys)