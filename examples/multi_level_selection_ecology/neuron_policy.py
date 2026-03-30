from abmax.structs import *
from abmax.functions import *
import jax.numpy as jnp
import jax.random as random
import jax
from sim_params import *

@struct.dataclass
class CTRNN(Policy):
    @staticmethod
    def create_policy(params:Params, key:jax.random.PRNGKey):
        num_neurons = params.content["num_neurons"]
        num_obs = params.content["num_obs"]
        num_actions = params.content["num_actions"]

        # initialization
        Z =  jnp.zeros((num_neurons,),dtype=jnp.float32)
        bar_Z = jnp.zeros((num_neurons,),dtype=jnp.float32)
        action = jnp.zeros((num_actions,),dtype=jnp.float32)
        
        key, *init_keys = jax.random.split(key, 6)
        J = jax.random.uniform(init_keys[0], shape = (num_neurons,num_neurons), minval=-1.0, maxval=1.0, dtype=jnp.float32) #interconnections
        E = jax.random.uniform(init_keys[1], shape = (num_neurons,num_obs), minval=-1.0, maxval=1.0, dtype=jnp.float32) # mapping from observations to neurons
        D = jax.random.uniform(init_keys[2], shape = (num_actions,num_neurons), minval=-1.0, maxval=1.0, dtype=jnp.float32) #readout
        tau = jax.random.uniform(init_keys[3], shape = (num_neurons,), minval=-1.0, maxval=1.0, dtype=jnp.float32) # time constants for each neuron
        B = jax.random.uniform(init_keys[4], shape = (num_neurons,), minval=-1.0, maxval=1.0, dtype=jnp.float32) # bias for each neuron
        state = State(content={'Z':Z, 'bar_Z':bar_Z, 'action':action})
        params = Params(content={'J':J, 'tau':tau, 'E':E, 'B':B, 'D':D})

        return CTRNN(params=params, state=state, key=key)
    
    @staticmethod
    @jax.jit
    def step_policy(policy:Policy, input:Signal, params:Params):
        dt = params.content['dt']
        action_scale = params.content['action_scale']
        time_constant_scale = params.content['time_constant_scale']

        J = policy.params.content['J']
        tau = policy.params.content['tau']
        E = policy.params.content['E']
        B = policy.params.content['B']
        D = policy.params.content['D']
        
        Z = policy.state.content['Z']
        bar_Z = policy.state.content['bar_Z']

        # get the input
        obs = input.content['obs']
        
        # compute the neuron updates
        z_dot = jnp.matmul(J, jax.nn.sigmoid(Z+B)) + jnp.matmul(E, obs) - Z
        #z_dot = jnp.tanh(jnp.matmul(J, Z) + jnp.matmul(E, obs) + B) - Z
        z_dot = jnp.multiply(z_dot, time_constant_scale*jax.nn.sigmoid(tau))

        new_Z = Z + dt*z_dot
        bar_Z = (1- BAR_NEURON_TIME_CONSTANT_SCALE) * bar_Z + BAR_NEURON_TIME_CONSTANT_SCALE * new_Z # update the bar neuron activity with smoothing

        #compute the action
        read_out = jnp.matmul(D, new_Z)
        actions = action_scale * jax.nn.tanh(read_out)

        new_policy_state = State(content={'Z':new_Z, 'bar_Z':bar_Z, 'action':actions})
        new_policy = policy.replace(state = new_policy_state)
        
        return new_policy
    
    @staticmethod
    @jax.jit
    def reset_policy(policy:Policy):
        Z = jnp.zeros_like(policy.state.content['Z'])
        bar_Z = jnp.zeros_like(policy.state.content['bar_Z'])
        action = jnp.zeros_like(policy.state.content['action'])
        
        new_policy_state = State(content={'Z':Z, 'bar_Z':bar_Z, 'action':action})
        new_policy = policy.replace(state = new_policy_state)
        
        return new_policy
    
    @staticmethod
    @jax.jit
    def set_policy(policy:Policy, set_params:Params):
        """
        Set the parameters of the policy to the given parameters
        Args:
            - policy: The policy to set
            - set_params: The parameters to set the policy to
        Returns:
            The updated policy
        """
        J = set_params.content['J']
        tau = set_params.content['tau']
        E = set_params.content['E']
        B = set_params.content['B']
        D = set_params.content['D']
        new_policy_params = Params(content={'J':J, 'tau':tau, 'E':E, 'B':B, 'D':D})
        return policy.replace(params = new_policy_params)



@struct.dataclass
class Selector_Network(Policy):
    @staticmethod
    def create_policy(key:jax.random.PRNGKey):
        key, *init_keys = jax.random.split(key, 6)
        
        w1 = jax.random.uniform(init_keys[0], shape = (MLP_SELECTOR_HIDDEN_SIZE, NUM_INPUTS_SELECTOR), minval=-1.0, maxval=1.0, dtype=jnp.float32) # hidden state 1 x input
        b1 = jax.random.uniform(init_keys[1], shape = (MLP_SELECTOR_HIDDEN_SIZE,), minval=-1.0, maxval=1.0, dtype=jnp.float32) # hidden state 1,
        w2 = jax.random.uniform(init_keys[2], shape = (MLP_SELECTOR_HIDDEN_SIZE, MLP_SELECTOR_HIDDEN_SIZE), minval=-1.0, maxval=1.0, dtype=jnp.float32) # hidden state 1 x hidden state 2
        b2 = jax.random.uniform(init_keys[3], shape = (MLP_SELECTOR_HIDDEN_SIZE,), minval=-1.0, maxval=1.0, dtype=jnp.float32) # hidden state 2
        w3 = jax.random.uniform(init_keys[4], shape = (MLP_SELECTOR_HIDDEN_SIZE, ), minval=-1.0, maxval=1.0, dtype=jnp.float32) # hidden state 2 x 1
        
        param_content = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2, 'w3': w3}
        params = Params(content=param_content)

        return Selector_Network(params=params, state=None, key=key)
    
    @staticmethod
    @jax.jit
    def step(policy:Policy, input:Signal, params:Params):
        w1 = policy.params.content['w1']
        b1 = policy.params.content['b1']
        w2 = policy.params.content['w2']
        b2 = policy.params.content['b2']
        w3 = policy.params.content['w3']

        x = input.content['input']
        hidden_1 = jnp.tanh(jnp.matmul(w1, x) + b1)
        hidden_2 = jnp.tanh(jnp.matmul(w2, hidden_1) + b2)
        output = jnp.tanh(jnp.matmul(w3, hidden_2))*MLP_SCALING # scale the output to keep it in a reasonable range for modulating the J values
        return output
    
    @staticmethod
    @jax.jit
    def set_policy(policy:Policy, set_params:Params):
        """
        Set the parameters of the policy to the given parameters
        Args:
            - policy: The policy to set
            - set_params: The parameters to set the policy to
        Returns:
            The updated policy
        """
        w1 = set_params.content['w1']
        b1 = set_params.content['b1']
        w2 = set_params.content['w2']
        b2 = set_params.content['b2']
        w3 = set_params.content['w3']
        new_policy_params = Params(content={'w1':w1, 'b1':b1, 'w2':w2, 'b2':b2, 'w3':w3})
        return policy.replace(params = new_policy_params)
    



if __name__ == "__main__":
    key = random.PRNGKey(0)
    selector_network = Selector_Network.create_policy(key)
    input = jnp.ones((NUM_INPUTS_SELECTOR,))
    output = Selector_Network.step(selector_network, Signal(content={'input': input}), None)
    print(output)