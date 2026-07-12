import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class NetworkTopology:
    """NNFacade - Dichiara l'hardware e il cablaggio"""
    n_neurons: int = struct.field(pytree_node=False)
    n_branches: int = struct.field(pytree_node=False)
    
    # Matrici di Adiacenza e Pesi [N_pre, N_post]
    w_exc: jnp.ndarray   # Pesi eccitatori
    w_inh: jnp.ndarray   # Pesi inibitori
    w_shunt: jnp.ndarray # Pesi shunting
    
    # Ritardi assonici (Dead Time in numero di scan)
    # Forma [N_pre, N_post] per permettere ritardi differenziali, oppure [N_pre] per un ritardo globale dell'assone
    axon_delays: jnp.ndarray 
    
    # Mappa dei rami [N_pre, N_post] -> int (da 0 a n_branches-1)
    # Indica su quale ramo del post-sinaptico si innesca la sinapsi
    branch_map: jnp.ndarray
    
    # Maschera di plasticità [N_pre, N_post]
    plasticity_mask: jnp.ndarray

def create_topology(n_neurons: int, n_branches: int) -> NetworkTopology:
    return NetworkTopology(
        n_neurons=n_neurons,
        n_branches=n_branches,
        w_exc=jnp.zeros((n_neurons, n_neurons), dtype=jnp.float32),
        w_inh=jnp.zeros((n_neurons, n_neurons), dtype=jnp.float32),
        w_shunt=jnp.zeros((n_neurons, n_neurons), dtype=jnp.float32),
        axon_delays=jnp.ones((n_neurons, n_neurons), dtype=jnp.int32),
        branch_map=jnp.zeros((n_neurons, n_neurons), dtype=jnp.int32),
        plasticity_mask=jnp.zeros((n_neurons, n_neurons), dtype=jnp.bool_)
    )
