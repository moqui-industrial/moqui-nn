import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple

@struct.dataclass
class SynapseState:
    """Stato Globale delle Sinapsi [N_pre, N_post]"""
    # Matrici dei Pesi (solo per eccitatori e inibitori plastici, ma possiamo tenere una generica e usare topo come maschera)
    weight: jnp.ndarray       # [N, N]
    
    # Short-Term Plasticity (STP)
    R: jnp.ndarray            # [N, N] Risorse Vescicolari (da 0.0 a 1.0)
    
    # Covariance Learning
    pre_rate: jnp.ndarray     # [N]
    post_rate: jnp.ndarray    # [N]
    pre_mean: jnp.ndarray     # [N]
    post_mean: jnp.ndarray    # [N]
    elig: jnp.ndarray         # [N, N]
    
    # Queuing Theory (Coda ad eventi discreti invece che pesi)
    buffer_level: jnp.ndarray # [N, N] Livello attuale del buffer (fessura sinaptica)

def init_synapse_state(w_initial: jnp.ndarray) -> SynapseState:
    n = w_initial.shape[0]
    return SynapseState(
        weight=w_initial,
        R=jnp.ones((n, n), dtype=jnp.float32),
        pre_rate=jnp.zeros(n, dtype=jnp.float32),
        post_rate=jnp.zeros(n, dtype=jnp.float32),
        pre_mean=jnp.zeros(n, dtype=jnp.float32),
        post_mean=jnp.zeros(n, dtype=jnp.float32),
        elig=jnp.zeros((n, n), dtype=jnp.float32),
        buffer_level=jnp.zeros((n, n), dtype=jnp.float32),
    )

def synapse_step(
    state: SynapseState,
    pre_spikes: jnp.ndarray,    # [N] array of bool
    post_spikes: jnp.ndarray,   # [N] array of bool
    plasticity_mask: jnp.ndarray, # [N, N] boolean mask (quali sinapsi possono imparare)
    dopamine: float,            # Globale
    # Hyperparams
    tau_rec: float = 15.0, U: float = 0.5,
    rate_tau: float = 12.0, mean_tau: float = 80.0,
    elig_leak: float = 0.92, lr: float = 0.5,
    w_min: float = 0.0, w_max: float = 20.0,
    QUANTUM: float = 150.0
) -> Tuple[SynapseState, jnp.ndarray]:
    """
    Esegue STP e Covarianza per tutte le connessioni NxN.
    Ritorna il nuovo stato e la matrice delle Cariche Efficaci Rilasciate [N_pre, N_post]
    """
    pre_f32 = pre_spikes.astype(jnp.float32)
    post_f32 = post_spikes.astype(jnp.float32)
    
    # ---------------------------------------------------------
    # 1. Short-Term Plasticity (STP)
    # ---------------------------------------------------------
    # Recupero continuo
    new_R = state.R + (1.0 - state.R) / tau_rec
    
    # Deplezione ai pre_spikes
    # Trasmettiamo il pre_spike alle righe della matrice R
    pre_matrix = jnp.expand_dims(pre_f32, axis=1) # [N, 1]
    
    # Il "peso" diventa la Capacità Massima del canale (recettori)
    channel_capacity = state.weight 
    
    # ---------------------------------------------------------
    # Step 2: Retrograde Signaling (Backpressure)
    # ---------------------------------------------------------
    # Se il buffer è saturo, il Dendrite invia un segnale di stop all'Assone.
    # fill_ratio indica quanto è pieno il buffer da 0.0 a 1.0
    fill_ratio = state.buffer_level / (channel_capacity * QUANTUM + 1e-5)
    backpressure = jnp.clip(fill_ratio, 0.0, 1.0)
    
    # L'Assone modula il rilascio leggendo SOLO la backpressure del Control Plane 
    # (Ripristinato il sistema DES PUSH veloce per i dati)
    release_fraction = U * new_R * (1.0 - backpressure)
    
    # Sottraiamo le risorse interne dell'assone solo per la frazione effettivamente rilasciata
    new_R = new_R - (release_fraction * pre_matrix)
    
    # L'assone spara i pacchetti sul canale (proporzionali alla capacità del canale/peso)
    PACKETS_PER_SPIKE = channel_capacity * QUANTUM 
    arriving_packets = release_fraction * pre_matrix * PACKETS_PER_SPIKE
    
    # Riempiamo la coda, scartando i pacchetti in eccesso (Buffer Overflow / Packet Loss)
    new_buffer_level = jnp.minimum(state.buffer_level + arriving_packets, channel_capacity * QUANTUM)
    
    # Il Dendrite consuma i pacchetti (Processing Rate / Inertia) proporzionalmente alla capacità
    PROCESSING_RATE = channel_capacity * QUANTUM # Tasso di svuotamento
    consumed_packets = jnp.minimum(new_buffer_level, PROCESSING_RATE)
    new_buffer_level = new_buffer_level - consumed_packets
    
    # La corrente effettiva è pari ai pacchetti processati dal Dendrite!
    effective_charge_matrix = consumed_packets
    
    # ---------------------------------------------------------
    # 2. Covariance Learning (Apprendimento Camma Elettronica)
    # ---------------------------------------------------------
    rate_alpha = 1.0 / rate_tau
    mean_alpha = 1.0 / mean_tau
    
    # Aggiornamento filtri 1D
    new_pre_rate = state.pre_rate + (pre_f32 - state.pre_rate) * rate_alpha
    new_post_rate = state.post_rate + (post_f32 - state.post_rate) * rate_alpha
    
    new_pre_mean = state.pre_mean + (new_pre_rate - state.pre_mean) * mean_alpha
    new_post_mean = state.post_mean + (new_post_rate - state.post_mean) * mean_alpha
    
    # Matrice delle coincidenze di fase [N, N]
    pre_dev = new_pre_rate - new_pre_mean # [N]
    post_dev = new_post_rate - new_post_mean # [N]
    
    phase_matrix = jnp.outer(pre_dev, post_dev) # [N_pre, N_post]
    
    # Pure Covariance (matching neuron_industrial.py)
    net_plasticity = phase_matrix
    
    # Eligibility Trace (matching neuron_industrial.py with direct addition)
    new_elig = state.elig * elig_leak + net_plasticity
    
    post_spikes_2d = jnp.expand_dims(post_f32, axis=0)    # [1, N_post]
    
    # Weight Update (Covarianza Hebbiana gated by post spikes, matching industrial reference)
    delta_w = new_elig * lr * dopamine * post_spikes_2d
    # Applichiamo il delta solo dove la plasticità è abilitata
    new_weight = state.weight + jnp.where(plasticity_mask, delta_w, 0.0)
    # Clip solo sui canali plastici (lasciando invariati i collegamenti riflessi cablati)
    new_weight = jnp.where(plasticity_mask, jnp.clip(new_weight, w_min, w_max), new_weight)
    
    # ---------------------------------------------------------
    # Normalizzazione Eterosinaptica (dei pesi plastici)
    # ---------------------------------------------------------
    # Somma dei pesi plastici in ingresso per ciascun neurone post-sinaptico [1, N_post]
    total_incoming_plastic = jnp.sum(jnp.where(plasticity_mask, new_weight, 0.0), axis=0, keepdims=True)
    # Calcolo fattore di riscalatura per non superare il budget totale
    max_total_capacity = 55.0
    scale = jnp.minimum(1.0, max_total_capacity / (total_incoming_plastic + 1e-5))
    # Riscaliamo solo i collegamenti plastici
    new_weight = jnp.where(plasticity_mask, new_weight * scale, new_weight)
    
    new_state = SynapseState(
        weight=new_weight,
        R=new_R,
        pre_rate=new_pre_rate,
        post_rate=new_post_rate,
        pre_mean=new_pre_mean,
        post_mean=new_post_mean,
        elig=new_elig,
        buffer_level=new_buffer_level
    )
    
    return new_state, effective_charge_matrix
