import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple

from moqui_nn.Neuron import NeuronState, init_neuron_state, neuron_step
from moqui_nn.Synapse import SynapseState, init_synapse_state, synapse_step
from moqui_nn.LoggerFacade import LogRingBuffer, init_log_ring_buffer, log_push, extract_log
from moqui_nn.NNFacade import NetworkTopology
from moqui_nn.Main import Blackboard

@struct.dataclass
class NetworkState:
    neurons: NeuronState
    synapses: SynapseState
    logger: LogRingBuffer
    spike_buffer: jnp.ndarray # Shape [MAX_DELAY, N]
    buffer_idx: jnp.int32
    time_step: jnp.int32
    axon_refractory_timer: jnp.ndarray # Shape [N]
    dopamine_buffer: jnp.ndarray # Shape [5]
    dopamine_idx: jnp.int32

def init_network_state(topology: NetworkTopology, max_delay: int, max_log_size: int = 1000) -> NetworkState:
    return NetworkState(
        neurons=init_neuron_state(topology.n_neurons, topology.n_branches),
        synapses=init_synapse_state(topology.w_exc), # Inizializziamo i pesi dinamici sulla matrice eccitatoria
        logger=init_log_ring_buffer(max_log_size, topology.n_neurons),
        spike_buffer=jnp.zeros((max_delay, topology.n_neurons), dtype=jnp.bool_),
        buffer_idx=jnp.int32(0),
        time_step=jnp.int32(0),
        axon_refractory_timer=jnp.zeros(topology.n_neurons, dtype=jnp.int32),
        dopamine_buffer=jnp.zeros(5, dtype=jnp.float32),
        dopamine_idx=jnp.int32(0)
    )

def network_step(state: NetworkState, topology: NetworkTopology, blackboard: Blackboard, external_exc: jnp.ndarray, external_inh: jnp.ndarray, disable_request: jnp.ndarray) -> Tuple[NetworkState, jnp.ndarray]:
    """
    Singolo Scan Cycle IEC 61131-3 vettorializzato in JAX.
    """
    n_neurons = topology.n_neurons
    
    # 1. Recupero Spikes ritardati dall'Axon Delay Buffer
    # Per semplicità in questa prima versione vettorializzata, usiamo un ritardo globale pre-calcolato.
    # Se topology.axon_delays fosse [N], faremmo un gather. 
    # Assumiamo per ora un delay fisso di 1 scan per tutti (lettura riga corrente del buffer).
    current_spikes = state.spike_buffer[state.buffer_idx] # Shape [N]
    
    # 2. Aggiornamento Sinapsi (STP e Covarianza con Ritardo Dopamina)
    # L'evento post-sinaptico per la covarianza (STDP) è il bAP (se il soma ha sparato allo scan PRECEDENTE)
    # Usiamo state.neurons.actuator.enable (spikes effettivi emessi dai somi al tempo t)
    post_spikes = state.neurons.actuator.enable
    
    # Canale di ritardo della dopamina (Event-Bookkeeping)
    DOPAMINE_DELAY = 5
    current_dopamine = state.dopamine_buffer[state.dopamine_idx]
    next_dopamine_idx = (state.dopamine_idx + 1) % DOPAMINE_DELAY
    new_dopamine_buffer = state.dopamine_buffer.at[next_dopamine_idx].set(blackboard.dopamine_level)
    
    new_synapses, effective_charge_matrix = synapse_step(
        state.synapses, 
        pre_spikes=current_spikes, 
        post_spikes=post_spikes,
        plasticity_mask=topology.plasticity_mask,
        dopamine=current_dopamine
    )
    
    # Sovrascriviamo la topologia con i pesi aggiornati dalla sinapsi (solo per la matrice eccitatoria)
    # N.B. In futuro potremmo estendere l'apprendimento alle matrici inibitorie.
    w_exc_dynamic = new_synapses.weight
    
    # 3. Routing Spikes (MatMul / Broadcasting)
    # effective_charge_matrix ha già calcolato (w_exc * STP_release_fraction * pre_spikes)
    # Dobbiamo solo sommare lungo l'asse pre-sinaptico [N_pre, N_post] -> [N_post],
    # e smistare nei branches.
    
    branch_mask = jax.nn.one_hot(topology.branch_map, topology.n_branches) # [N_pre, N_post, n_branches]
    effective_charge_3d = jnp.expand_dims(effective_charge_matrix, axis=-1) * branch_mask # [N_pre, N_post, n_branches]
    
    # i_exc_branches: [N_post, n_branches]
    i_exc_branches = jnp.sum(effective_charge_3d, axis=0)
    
    # Aggiungiamo input esterno (assumiamo vada sul branch 0)
    external_exc_branches = jnp.zeros_like(i_exc_branches).at[:, 0].set(external_exc)
    total_i_exc = i_exc_branches + external_exc_branches
    
    # Per inibitorio e shunting, usiamo i pesi statici (niente STP/Covarianza per ora)
    spike_f32 = current_spikes.astype(jnp.float32)
    total_i_inh = jnp.dot(spike_f32, topology.w_inh) + external_inh
    total_i_shunt = jnp.dot(spike_f32, topology.w_shunt)
    
    # 4. Aggiornamento Neuroni
    # vmap trasforma una funzione operante su 1 neurone in una operante su N neuroni
    # siccome neuron_step prende intere ndarray e fa math globale (perchè l'abbiamo scritta così in Neuron.py), 
    # non serve neanche vmap! neuron_step usa GIA' le shape [N].
    
    new_neurons = neuron_step(state.neurons, total_i_exc, total_i_inh, total_i_shunt, disable_request)
    
    # 4. Aggiornamento Axon Delay Buffer
    new_spikes = new_neurons.actuator.enable
    
    # Axonal Refractoriness / Packet Drop due to congestion
    # Se il timer assonale è attivo (>0), lo spike viene perso (packet drop stocastico/refrattario)
    axon_is_refractory = state.axon_refractory_timer > 0
    effective_spikes = new_spikes & ~axon_is_refractory
    
    # Aggiornamento timer refrattario assonale: se c'è spike effettivo imposta a 2, altrimenti decrementa
    new_axon_timer = jnp.where(effective_spikes, 2, jnp.maximum(0, state.axon_refractory_timer - 1))
    
    max_delay = state.spike_buffer.shape[0]
    next_idx = (state.buffer_idx + 1) % max_delay
    
    new_spike_buffer = state.spike_buffer.at[next_idx].set(effective_spikes)
    
    # 5. Logging
    # Estrarre status = 2 (STATUS_ENABLED) non è necessario qui, salviamo tutta l'FSM
    # Ma per l'addestramento offline ci interessa se spara. 
    # Salviamo l'intera FSM as byte.
    new_status_word = new_neurons.actuator.status.astype(jnp.uint8)
    new_logger = log_push(state.logger, new_status_word)
    
    new_state = NetworkState(
        neurons=new_neurons,
        synapses=new_synapses,
        logger=new_logger,
        spike_buffer=new_spike_buffer,
        buffer_idx=next_idx,
        time_step=state.time_step + 1,
        axon_refractory_timer=new_axon_timer,
        dopamine_buffer=new_dopamine_buffer,
        dopamine_idx=next_dopamine_idx
    )
    
    return new_state, effective_spikes

def run_network_scan(initial_state: NetworkState, topology: NetworkTopology, blackboard: Blackboard, ext_exc_seq: jnp.ndarray, ext_inh_seq: jnp.ndarray, disable_seq: jnp.ndarray):
    """
    Esegue M scansioni della rete.
    ext_exc_seq ha shape [M, N]
    Restituisce (stato_finale, spike_history [M, N])
    """
    def scan_fn(state, inputs):
        ext_exc, ext_inh, disable = inputs
        new_state, out_spikes = network_step(state, topology, blackboard, ext_exc, ext_inh, disable)
        return new_state, out_spikes
    
    final_state, spike_history = jax.lax.scan(scan_fn, initial_state, (ext_exc_seq, ext_inh_seq, disable_seq))
    return final_state, spike_history

def replay_log(state: NetworkState, topology: NetworkTopology, blackboard: Blackboard) -> NetworkState:
    """
    Sleep Mode: Riproduce le memorie registrate nel LogRingBuffer per consolidare la plasticità offline.
    Inietta i log passati nel motore sinaptico ad altissima velocità.
    """
    extracted_log = extract_log(state.logger) # [LOG_SIZE, N] di uint8 (status word)
    
    # Gli eventi di spike sono quelli in cui status == 2 (STATUS_ENABLED)
    # STATUS_ENABLED = 2 (definito implicitamente nella logica)
    spikes_log = (extracted_log == 2) # [LOG_SIZE, N] di bool
    
    # Creiamo sequenze pre_spikes (t) e post_spikes (t-1)
    pre_spikes_seq = spikes_log[1:]
    post_spikes_seq = spikes_log[:-1]
    
    def replay_scan_fn(syn_state: SynapseState, inputs):
        pre, post = inputs
        new_syn_state, _ = synapse_step(
            syn_state,
            pre_spikes=pre,
            post_spikes=post,
            plasticity_mask=topology.plasticity_mask,
            dopamine=blackboard.dopamine_level
        )
        return new_syn_state, None
        
    final_synapses, _ = jax.lax.scan(replay_scan_fn, state.synapses, (pre_spikes_seq, post_spikes_seq))
    
    return state.replace(synapses=final_synapses)

