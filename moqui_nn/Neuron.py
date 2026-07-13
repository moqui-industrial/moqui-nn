import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple

from moqui_nn.Constants import *

@struct.dataclass
class ActuatorState:
    status: jnp.ndarray
    spike_timer: jnp.ndarray
    refractory_timer: jnp.ndarray
    enable: jnp.ndarray

@struct.dataclass
class DendriteState:
    accumulator: jnp.ndarray
    actual_rate: jnp.ndarray
    hungry: jnp.ndarray
    branch_exc: jnp.ndarray
    fatigue: jnp.ndarray      # Thermal Throttling / SFA

@struct.dataclass
class NeuronState:
    actuator: ActuatorState
    dendrite: DendriteState

# --- Pure Functions (IEC 61131-3 Block Logic) ---

def init_neuron_state(n_neurons: int, n_branches: int) -> NeuronState:
    return NeuronState(
        actuator=ActuatorState(
            status=jnp.zeros(n_neurons, dtype=jnp.int32),
            spike_timer=jnp.zeros(n_neurons, dtype=jnp.int32),
            refractory_timer=jnp.zeros(n_neurons, dtype=jnp.int32),
            enable=jnp.zeros(n_neurons, dtype=jnp.bool_),
        ),
        dendrite=DendriteState(
            accumulator=jnp.zeros(n_neurons, dtype=jnp.float32),
            actual_rate=jnp.zeros(n_neurons, dtype=jnp.float32),
            hungry=jnp.zeros(n_neurons, dtype=jnp.bool_),
            branch_exc=jnp.zeros((n_neurons, n_branches), dtype=jnp.float32),
            fatigue=jnp.zeros(n_neurons, dtype=jnp.float32)
        )
    )

def actuator_step(state: ActuatorState, enable_request: jnp.ndarray, disable_request: jnp.ndarray, spike_width: int, refractory: int) -> ActuatorState:
    # 1. Disable Request asincrono ESTERNO (ha la precedenza): un comando da un
    # livello di orchestrazione sopra il soma puo' sempre forzare la disabilitazione
    # immediatamente, qualunque sia la fase corrente.
    status_init = jnp.where(disable_request, STATUS_DISABLE_PHASE, state.status)
    refractory_timer_init = jnp.where(disable_request, refractory, state.refractory_timer)
    spike_timer_init = state.spike_timer

    # 2. Condizioni basate sullo stato di inizio scansione (snapshot)
    cond_disabled = (status_init == STATUS_DISABLED)
    cond_enable_phase = (status_init == STATUS_ENABLE_PHASE)
    cond_enabled = (status_init == STATUS_ENABLED)
    cond_disable_phase = (status_init == STATUS_DISABLE_PHASE)

    # 3. Transizioni
    to_enable = cond_disabled & enable_request

    # Il timer conta sia la rampa di ingresso (EnablePhase, tempo per raggiungere
    # Enabled) sia la durata della depolarizzazione (Enabled) - due fasi consecutive,
    # mai simultanee, quindi possono condividere lo stesso contatore.
    dec_timer = (cond_enable_phase | cond_enabled) & (spike_timer_init > 0)
    next_spike_timer = jnp.where(dec_timer, spike_timer_init - 1, spike_timer_init)
    next_spike_timer = jnp.where(to_enable, spike_width, next_spike_timer)

    to_enabled = cond_enable_phase & (next_spike_timer == 0)
    # Al passaggio a Enabled si riarma il timer per contare la durata della
    # depolarizzazione, prima di autoemettere la richiesta di disabilitazione.
    next_spike_timer = jnp.where(to_enabled, spike_width, next_spike_timer)

    # Disable Request INTERNO: il soma se lo emette da solo quando il proprio tempo
    # di depolarizzazione e' scaduto. La ripolarizzazione di un vero potenziale
    # d'azione e' autonoma (nessun controllore esterno la comanda), ma passa dallo
    # stesso canale control-plane della richiesta esterna (gia' gestita al punto 1)
    # invece che da un salto di stato diretto che scavalca l'handshake: 'Enabled'
    # resta uno stato stabile finche' non arriva un disableRequest, esterno o
    # interno che sia - esattamente come nell'Actuator originale (Actuator.st),
    # non un timeout incondizionato dentro la sola FSM del soma.
    internal_disable_request = cond_enabled & (next_spike_timer == 0)
    to_disable_phase = cond_enabled & internal_disable_request

    new_enable = cond_enabled

    dec_refractory_timer = cond_disable_phase & (refractory_timer_init > 0)
    next_refractory_timer = jnp.where(dec_refractory_timer, refractory_timer_init - 1, refractory_timer_init)
    next_refractory_timer = jnp.where(to_disable_phase, refractory, next_refractory_timer)

    to_disabled = cond_disable_phase & (next_refractory_timer == 0)

    # 4. Assegnazione nuovo stato
    new_status = status_init
    new_status = jnp.where(to_enable, STATUS_ENABLE_PHASE, new_status)
    new_status = jnp.where(to_enabled, STATUS_ENABLED, new_status)
    new_status = jnp.where(to_disable_phase, STATUS_DISABLE_PHASE, new_status)
    new_status = jnp.where(to_disabled, STATUS_DISABLED, new_status)

    return ActuatorState(
        status=new_status,
        spike_timer=next_spike_timer,
        refractory_timer=next_refractory_timer,
        enable=new_enable
    )

def dendrite_step(
    state: DendriteState, 
    i_exc: jnp.ndarray,    # Shape [N, n_branches]
    i_inh: jnp.ndarray,    # Shape [N]
    i_shunt: jnp.ndarray,  # Shape [N]
    bap: jnp.ndarray,      # Back-Action Potential (se il soma ha sparato)
    threshold: float=1000.0, ceiling: float=1700.0, floor: float=-700.0,
    leak_rate: float=120.0, local_th: float=1200.0, local_boost: float=1500.0,
    set_rate: float=0.10, rate_tau: float=20.0, hungry_drop: float=350.0,
    fatigue_decay: float=15.0, fatigue_bump: float=500.0, fatigue_capacity: float=1500.0
) -> Tuple[DendriteState, jnp.ndarray]:
    
    # Reset accumulator se bAP
    acc = jnp.where(bap, 0.0, state.accumulator)
    
    # Leakage
    acc = jnp.where(acc > 0.0, jnp.maximum(0.0, acc - leak_rate), acc)
    acc = jnp.where(acc < 0.0, jnp.minimum(0.0, acc + leak_rate), acc)
    
    # Branch non-linear summation
    total_exc_branches = state.branch_exc + i_exc
    boosted_branches = jnp.where(total_exc_branches > local_th, total_exc_branches + local_boost, total_exc_branches)
    total_exc = jnp.sum(boosted_branches, axis=1) # Somma lungo i rami
    
    # Integration
    acc = jnp.minimum(ceiling, acc + total_exc)
    acc = jnp.maximum(floor, acc - i_inh)
    
    # Shunting
    acc = jnp.where(i_shunt > 0, acc / (1.0 + i_shunt / 1000.0), acc)
    
    # ---------------------------------------------------------
    # Step 5: Thermal Throttling (Spike Frequency Adaptation)
    # ---------------------------------------------------------
    # Coda a capacita' e tempo di servizio finiti: decade di fatigue_decay a ogni
    # ciclo (tempo di servizio finito, gia' cosi' in origine), e si ricarica di
    # fatigue_bump a ogni spike MA con un tetto fatigue_capacity - senza il tetto,
    # una sequenza di spike ravvicinati farebbe crescere la fatica senza limite,
    # cosa che nessun meccanismo di adattamento reale (canali K+ Ca-dipendenti,
    # che saturano) fa.
    new_fatigue = jnp.maximum(0.0, state.fatigue - fatigue_decay)
    new_fatigue = jnp.where(bap, jnp.minimum(fatigue_capacity, new_fatigue + fatigue_bump), new_fatigue)
    
    # Enable Request (Hungry Homeostasis + Thermal Throttling)
    # La soglia sale per la fatica, ma scende se è affamato
    eff_threshold = threshold + new_fatigue - jnp.where(state.hungry, hungry_drop, 0.0)
    enable_request = acc >= eff_threshold
    
    # Homeostasis update
    new_actual_rate = state.actual_rate + (enable_request.astype(jnp.float32) - state.actual_rate) / rate_tau
    new_hungry = new_actual_rate < set_rate
    
    new_state = DendriteState(
        accumulator=acc,
        actual_rate=new_actual_rate,
        hungry=new_hungry,
        branch_exc=jnp.zeros_like(state.branch_exc), # Consumato al tick
        fatigue=new_fatigue
    )
    
    return new_state, enable_request

def neuron_step(state: NeuronState, i_exc: jnp.ndarray, i_inh: jnp.ndarray, i_shunt: jnp.ndarray, disable_request: jnp.ndarray) -> NeuronState:
    # 1. Dendrite elabora ingressi
    new_dendrite, enable_request = dendrite_step(
        state.dendrite, i_exc, i_inh, i_shunt, bap=state.actuator.enable
    )

    # 2. Soma riceve enable_request
    new_actuator = actuator_step(state.actuator, enable_request, disable_request, spike_width=1, refractory=3)

    return NeuronState(actuator=new_actuator, dendrite=new_dendrite)
