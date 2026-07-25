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
    dominance_timer: jnp.ndarray # Shape [N] - coda a tempo di servizio finito (vedi network_step)

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
        dopamine_idx=jnp.int32(0),
        dominance_timer=jnp.zeros(topology.n_neurons, dtype=jnp.int32)
    )

def network_step(state: NetworkState, topology: NetworkTopology, blackboard: Blackboard, external_exc: jnp.ndarray, external_inh: jnp.ndarray, disable_request: jnp.ndarray, external_target: jnp.ndarray = None) -> Tuple[NetworkState, jnp.ndarray]:
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
        dopamine=current_dopamine,
        max_total_capacity=topology.max_synaptic_budget,
        max_presynaptic_capacity=topology.max_presynaptic_capacity,
        homeostatic_tau=topology.homeostatic_tau,
        target_signal=external_target,
        use_delta_rule=topology.use_delta_rule,
        q_capacity=topology.delta_rule_q_capacity,
        ki_ts=topology.delta_rule_ki_ts,
        decay_period=topology.delta_rule_decay_period,
        decay_quantum=topology.delta_rule_decay_quantum
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
    
    # Gettone di dominanza (fatto contabile: "ho vinto entro le ultime DOMINANCE_HOLD
    # scansioni"), usato sia per l'inibizione competitiva sotto sia per l'attenzione
    # top-down qui sopra: e' lo stesso segnale, letto in due direzioni diverse.
    is_dominant = state.dominance_timer > 0
    dominant_f32 = is_dominant.astype(jnp.float32)

    # Attenzione top-down: un motore dominante inietta corrente eccitatoria
    # all'indietro sui presinaptici da cui dipende di piu' - stesso filo del peso
    # gia' imparato (w_exc_dynamic), percorso al contrario, non una matrice nuova.
    # Attiva solo fuori dal training (dopamina <= 0): durante l'apprendimento il
    # teacher forcing guida gia' il motore giusto, e aggiungere qui un rinforzo
    # rischierebbe di far saturare i pesi ancora piu' in fretta.
    attention_active = (blackboard.dopamine_level <= 0.0).astype(jnp.float32)
    attention_current = topology.attention_gain * attention_active * jnp.dot(w_exc_dynamic, dominant_f32)

    # Aggiungiamo input esterno (assumiamo vada sul branch 0) e l'eventuale
    # rinforzo top-down (stesso branch: e' comunque corrente eccitatoria in ingresso)
    external_exc_branches = jnp.zeros_like(i_exc_branches).at[:, 0].add(external_exc + attention_current)
    total_i_exc = i_exc_branches + external_exc_branches

    # Per inibitorio e shunting, usiamo i pesi statici (niente STP/Covarianza per ora)
    # Inibizione competitiva pilotata dal GETTONE DI DOMINANZA (fatto contabile:
    # "ho vinto entro le ultime DOMINANCE_HOLD scansioni"), non dallo spike istantaneo.
    # Lo spike grezzo dura 1 solo ciclo: usarlo direttamente come sorgente di inibizione
    # produce un carosello a rotazione (il vincitore va in refrattario, l'inibizione
    # sparisce con lui, il prossimo in classifica ha via libera). Il gettone e' una coda
    # a tempo di servizio finito, esattamente come axon_refractory_timer: si ricarica al
    # valore pieno a ogni evento di vittoria, scarica di 1 a ogni ciclo, ed e' "vero" finche'
    # non si esaurisce - un fatto contabile discreto, non un confronto su un livello continuo.
    #
    # Tutta la rete legge qui lo stesso stato congelato di 'state' (dalla fine del ciclo
    # precedente) e scrive in 'new_neurons', un oggetto nuovo, mai in-place - esattamente
    # come InputSignalUpdate->Main->OutputSignalUpdate nel vero PLC: ingressi congelati per
    # l'intero scan, commit unico a fine ciclo. L'ordine (qui: nessun ordine, e' un'unica
    # operazione vettorializzata) non conta perche' nessun neurone vede aggiornamenti "in
    # corso" di un altro entro lo stesso ciclo - se piu' neuroni pareggiano, e' un sintomo
    # onesto di pesi non ancora sufficientemente differenziati, non un difetto da mascherare
    # con un ordine di esecuzione arbitrario.
    total_i_inh = jnp.dot(dominant_f32, topology.w_inh) + external_inh
    total_i_shunt = jnp.dot(dominant_f32, topology.w_shunt)

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

    # Aggiornamento del gettone di dominanza: la ricarica NON e' un valore fisso uguale
    # per ogni vincitore, altrimenti l'intera coorte di chi supera la soglia quasi insieme
    # si inibirebbe a vicenda e scadrebbe tutta insieme (carosello piu' lento, ma sempre
    # carosello). La ricarica e' 1 ciclo garantito per qualunque vittoria (altrimenti un
    # margine di sovrasoglia sotto un quanto arrotonderebbe a zero, come infatti accadeva:
    # il tetto configurato non aveva alcun effetto sui vincitori marginali) PIU' un margine
    # aggiuntivo proporzionale a QUANTI pacchetti di sovrasoglia il neurone ha accumulato al
    # momento della vittoria (stessa convenzione a pacchetti gia' usata per il canale
    # sinaptico in Synapse.py - un fatto contabile, non una formula continua), con tetto a
    # topology.dominance_hold (la capacita' finita della coda). Con dominance_hold=1
    # (default) il tetto riporta sempre a 1: nessun cambiamento per chi non lo configura.
    DOMINANCE_QUANTUM = 100.0
    surplus_packets = jnp.maximum(0.0, new_neurons.dendrite.accumulator - 1000.0) / DOMINANCE_QUANTUM
    token_gain = jnp.minimum(topology.dominance_hold, 1 + surplus_packets.astype(jnp.int32))
    new_dominance_timer = jnp.where(effective_spikes, jnp.maximum(state.dominance_timer, token_gain), jnp.maximum(0, state.dominance_timer - 1))

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
        dopamine_idx=next_dopamine_idx,
        dominance_timer=new_dominance_timer
    )
    
    return new_state, effective_spikes

def run_network_scan(initial_state: NetworkState, topology: NetworkTopology, blackboard: Blackboard, ext_exc_seq: jnp.ndarray, ext_inh_seq: jnp.ndarray, disable_seq: jnp.ndarray, ext_target_seq: jnp.ndarray = None):
    """
    Esegue M scansioni della rete.
    ext_exc_seq ha shape [M, N]
    Restituisce (stato_finale, spike_history [M, N])
    """
    if ext_target_seq is None:
        # Nessun bersaglio esterno fornito: la regola delta locale (se mai attiva
        # per questa topologia) non riceve mai un evento "bersaglio attivo".
        ext_target_seq = jnp.zeros_like(ext_exc_seq, dtype=jnp.bool_)

    def scan_fn(state, inputs):
        ext_exc, ext_inh, disable, ext_target = inputs
        new_state, out_spikes = network_step(state, topology, blackboard, ext_exc, ext_inh, disable, ext_target)
        return new_state, out_spikes

    final_state, spike_history = jax.lax.scan(scan_fn, initial_state, (ext_exc_seq, ext_inh_seq, disable_seq, ext_target_seq))
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
            dopamine=blackboard.dopamine_level,
            max_total_capacity=topology.max_synaptic_budget,
            homeostatic_tau=topology.homeostatic_tau
        )
        return new_syn_state, None
        
    final_synapses, _ = jax.lax.scan(replay_scan_fn, state.synapses, (pre_spikes_seq, post_spikes_seq))
    
    return state.replace(synapses=final_synapses)

