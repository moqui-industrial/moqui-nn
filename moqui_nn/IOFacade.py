import jax
import jax.numpy as jnp
from flax import struct
from typing import Dict, List

def map_sensory_inputs(odor_intensity: float, sensory_indices: jnp.ndarray, n_neurons: int) -> jnp.ndarray:
    """
    Traduce il livello di odore in corrente eccitatoria (drive) da iniettare nei sensori.
    """
    ext_exc = jnp.zeros(n_neurons, dtype=jnp.float32)
    # L'odore diventa corrente proporzionale
    drive = odor_intensity * 100.0 # Es. se odor è 10.0, drive è 1000.0 (soglia di sparo)
    # Applichiamo il drive a tutti gli indici sensoriali forniti
    
    # In JAX jnp.zeros non supporta l'assegnazione diretta. Usiamo .at[].set()
    ext_exc = ext_exc.at[sensory_indices].set(drive)
    return ext_exc

def map_motor_outputs(spikes: jnp.ndarray, forward_indices: jnp.ndarray, turn_left_indices: jnp.ndarray, turn_right_indices: jnp.ndarray) -> jnp.ndarray:
    """
    Legge gli spike dai motoneuroni e restituisce [forward_thrust, turn_thrust].
    """
    # Contiamo quanti spike ci sono in ogni gruppo
    forward_count = jnp.sum(spikes[forward_indices])
    left_count = jnp.sum(spikes[turn_left_indices])
    right_count = jnp.sum(spikes[turn_right_indices])
    
    forward_thrust = forward_count * 0.5
    turn_thrust = (right_count - left_count) * 0.2
    
    return jnp.array([forward_thrust, turn_thrust])

def get_celegans_io_indices(node_names: Dict[int, str]) -> tuple:
    """
    Ritorna gli array JAX degli indici per Sensori e Motori noti del C. Elegans.
    """
    sensory = []
    forward = []
    turn_left = []
    turn_right = []
    
    for idx, name in node_names.items():
        # Sensori Chemotattici (Amphid)
        if name in ['ASEL', 'ASER', 'AWAL', 'AWAR', 'AWCL', 'AWCR']:
            sensory.append(idx)
        # Motori (Ventral/Dorsal Body per avanzamento)
        elif name.startswith('VB') or name.startswith('DB'):
            forward.append(idx)
        # Motori Sterzanti (Head/Neck)
        elif name in ['SMBVL', 'SMBDL', 'RMDVL', 'RMDL']: # Sinistra
            turn_left.append(idx)
        elif name in ['SMBVR', 'SMBDR', 'RMDVR', 'RMDR']: # Destra
            turn_right.append(idx)
            
    return (
        jnp.array(sensory, dtype=jnp.int32),
        jnp.array(forward, dtype=jnp.int32),
        jnp.array(turn_left, dtype=jnp.int32),
        jnp.array(turn_right, dtype=jnp.int32)
    )
