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

    # Regola Delta Locale (azione integrale su code, alternativa opt-in alla
    # covarianza pura): due code a capacita' finita per neurone post-sinaptico,
    # alimentate da eventi discreti (sparo si'/no, bersaglio si'/no), non da tassi.
    q_current: jnp.ndarray  # [N] quanto il neurone ha effettivamente sparato di recente
    q_target: jnp.ndarray   # [N] quanto il bersaglio esterno chiede che sia attivo

    # Stagionatura per-sinapsi: cicli trascorsi dall'ultimo evento di rinforzo/
    # punizione su QUELLA specifica connessione (stessa natura di refractory_timer,
    # ma per connessione invece che per neurone). Confrontata contro una soglia
    # intera (come tempOverMax := T >= TMax), non moltiplicata in una formula.
    staleness: jnp.ndarray  # [N, N]

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
        q_current=jnp.zeros(n, dtype=jnp.float32),
        q_target=jnp.zeros(n, dtype=jnp.float32),
        staleness=jnp.zeros((n, n), dtype=jnp.int32),
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
    QUANTUM: float = 150.0, max_total_capacity: float = 55.0,
    homeostatic_tau: float = 1.0,
    target_signal: jnp.ndarray = None, use_delta_rule: bool = False,
    q_capacity: float = 10.0, ki_ts: float = 1.0,
    decay_period: int = 50, decay_quantum: float = 0.0
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

    # ---------------------------------------------------------
    # Regola Delta Locale (azione integrale su code, opt-in via use_delta_rule)
    # ---------------------------------------------------------
    # Due code a capacita' finita per neurone post-sinaptico, alimentate da fatti
    # discreti (ho sparato? il bersaglio lo chiede?), non da tassi continui.
    new_q_current = jnp.where(post_spikes, jnp.minimum(q_capacity, state.q_current + 1.0),
                               jnp.maximum(0.0, state.q_current - 1.0))
    target_bool = target_signal if target_signal is not None else jnp.zeros_like(state.q_target, dtype=jnp.bool_)
    new_q_target = jnp.where(target_bool, jnp.minimum(q_capacity, state.q_target + 1.0),
                              jnp.maximum(0.0, state.q_target - 1.0))
    # errore = differenza intera di due code (non una sottrazione di tassi continui)
    error = new_q_target - new_q_current  # [N_post]
    # Azione integrale: il peso stesso e' la coda a capacita' finita (w_min/w_max).
    # Un quanto fisso (Ki*Ts) per errore, per ogni ciclo (Ts=1 scansione, implicito),
    # solo dove il presinaptico e' idoneo (ha sparato in questo stesso ciclo) - e
    # solo quando l'apprendimento e' abilitato (gate discreto sulla dopamina, non
    # una moltiplicazione continua).
    learning_gate = (dopamine > 0.0).astype(jnp.float32)
    delta_w_local = ki_ts * learning_gate * jnp.expand_dims(error, axis=0) * jnp.expand_dims(pre_f32, axis=1)

    # use_delta_rule arriva da un campo di topologia (pytree_node=False): e' un
    # valore Python statico, non tracciato - una diramazione Python normale, non
    # jnp.where, cosi' evitiamo di calcolare il ramo non usato.
    if use_delta_rule:
        delta_w = delta_w_local

    # Stagionatura per-sinapsi (non una fuga percentuale continua): ogni sinapsi
    # conta quanti cicli sono passati dal suo ultimo evento di rinforzo/punizione
    # (azzerata a ogni evento reale, altrimenti +1 a ciclo - stessa natura di
    # refractory_timer). Quando il contatore supera una soglia intera (confronto,
    # come tempOverMax := T >= TMax, non una formula), si scarica UN pacchetto
    # fisso di peso (decay_quantum), non una frazione. Senza questa stagionatura,
    # una manopola che raggiunge il tetto o insegue il proprio bersaglio fino a
    # errore zero ci resta bloccata per sempre (windup di un integratore puro):
    # non distingue piu' "rinforzata di continuo" da "rinforzata una volta e mai
    # piu' toccata". Con decay_quantum=0.0 (default) e' un no-op.
    had_event = jnp.where(plasticity_mask, delta_w != 0.0, False)
    new_staleness = jnp.where(had_event, 0, state.staleness + 1)
    is_stale = new_staleness >= decay_period
    new_staleness = jnp.where(is_stale, 0, new_staleness)
    decay_amount = jnp.where(is_stale & plasticity_mask, decay_quantum, 0.0)
    base_weight = jnp.maximum(w_min, state.weight - decay_amount)

    # Applichiamo il delta solo dove la plasticità è abilitata
    new_weight = base_weight + jnp.where(plasticity_mask, delta_w, 0.0)
    # Clip solo sui canali plastici (lasciando invariati i collegamenti riflessi cablati)
    new_weight = jnp.where(plasticity_mask, jnp.clip(new_weight, w_min, w_max), new_weight)
    
    # ---------------------------------------------------------
    # Normalizzazione Eterosinaptica (dei pesi plastici)
    # ---------------------------------------------------------
    # Somma dei pesi plastici in ingresso per ciascun neurone post-sinaptico [1, N_post]
    total_incoming_plastic = jnp.sum(jnp.where(plasticity_mask, new_weight, 0.0), axis=0, keepdims=True)
    # Calcolo fattore di riscalatura per non superare il budget totale
    # (il budget e' un parametro dell'impianto: dipende dal fan-in reale del
    # neurone post-sinaptico, non una costante universale)
    target_scale = jnp.minimum(1.0, max_total_capacity / (total_incoming_plastic + 1e-5))
    # Lo scaling omeostatico reale (synaptic scaling) e' un processo lento (ore/
    # giorni via traffico dei recettori), non un ricalcolo istantaneo a ogni ciclo.
    # homeostatic_tau=1.0 (default) applica la correzione per intero in un ciclo,
    # identico al comportamento originale; valori piu' alti la applicano solo in
    # parte a ogni ciclo, avvicinandosi al budget gradualmente su molti cicli.
    scale = 1.0 + (target_scale - 1.0) / homeostatic_tau
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
        buffer_level=new_buffer_level,
        q_current=new_q_current,
        q_target=new_q_target,
        staleness=new_staleness
    )
    
    return new_state, effective_charge_matrix
