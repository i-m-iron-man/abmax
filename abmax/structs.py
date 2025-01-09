from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class Signal:
    content:dict

@struct.dataclass
class Params:
    content:dict

@struct.dataclass
class State:
    content:dict

@struct.dataclass
class Policy:
    state:State
    params:Params
    key: jax.random.PRNGKey


@struct.dataclass
class Agent:
    id:jnp.int32
    active_state:bool
    age:jnp.float32
    agent_type:jnp.int32
    
    params:Params
    state:State
    policy:Policy
    key: jax.random.PRNGKey

@struct.dataclass
class Set:
    
    num_agents:jnp.int32
    num_active_agents:jnp.int32
    agents:Agent

    id:jnp.int32
    set_type:jnp.int32
    
    params:Params
    state:State
    policy:Policy
    key: jax.random.PRNGKey
    