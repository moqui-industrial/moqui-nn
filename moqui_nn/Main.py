import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class Blackboard:
    """Stato dell'organismo/ecosistema (Neuromodulatori globali)."""
    dopamine_level: float
    serotonin_level: float
    sleep_mode: bool

def ecosystem_update(blackboard: Blackboard, global_reward_signal: float) -> Blackboard:
    """
    MainRuleEngine.st - L'orchestratore di alto livello (eseguito a frequenza minore).
    Regola i neuromodulatori in base al feedback dell'ambiente.
    """
    new_da = blackboard.dopamine_level + global_reward_signal * 0.1
    new_da = jnp.clip(new_da, 0.0, 1.0)
    
    return Blackboard(
        dopamine_level=new_da,
        serotonin_level=blackboard.serotonin_level,
        sleep_mode=blackboard.sleep_mode
    )
