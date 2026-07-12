import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class EnvironmentState:
    worm_x: float
    worm_y: float
    worm_theta: float
    food_x: float
    food_y: float
    # Valore di odore calcolato nello step precedente (utile per misurare il gradiente)
    last_odor: float

def init_environment(food_x: float, food_y: float) -> EnvironmentState:
    return EnvironmentState(
        worm_x=0.0,
        worm_y=0.0,
        worm_theta=0.0,
        food_x=food_x,
        food_y=food_y,
        last_odor=0.0
    )

def environment_step(state: EnvironmentState, forward_thrust: float, turn_thrust: float, dt: float = 0.1) -> EnvironmentState:
    """
    Muove il verme nell'ambiente 2D in base alla trazione propulsiva e sterzante.
    """
    new_theta = state.worm_theta + turn_thrust * dt
    
    new_x = state.worm_x + forward_thrust * jnp.cos(new_theta) * dt
    new_y = state.worm_y + forward_thrust * jnp.sin(new_theta) * dt
    
    # Calcolo dell'odore (inversamente proporzionale alla distanza dal cibo)
    dist = jnp.sqrt((new_x - state.food_x)**2 + (new_y - state.food_y)**2)
    new_odor = 100.0 / (1.0 + dist)
    
    return EnvironmentState(
        worm_x=new_x,
        worm_y=new_y,
        worm_theta=new_theta,
        food_x=state.food_x,
        food_y=state.food_y,
        last_odor=new_odor
    )
