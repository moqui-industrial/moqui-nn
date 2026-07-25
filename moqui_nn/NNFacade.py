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

    # Tempo di mantenimento del gettone di dominanza, per neurone [N] (in scansioni).
    # E' un parametro fisico di popolazione (come enableTime/disableTime, che variano
    # per modello di attuatore nel framework reale): 1 = comportamento originale
    # spike-triggered (il gettone dura esattamente 1 ciclo, indistinguibile dallo spike
    # grezzo); valori piu' alti fanno persistere l'inibizione competitiva di quel neurone
    # oltre il proprio refrattario, per popolazioni che devono stabilizzare un codice
    # sparso duraturo invece di una decisione istantanea.
    dominance_hold: jnp.ndarray

    # Budget di normalizzazione eterosinaptica per neurone post-sinaptico.
    # Dipende dal fan-in reale dell'impianto (quanti presinaptici plastici
    # convergono), quindi e' un parametro di topologia, non una costante fissa.
    max_synaptic_budget: float = struct.field(pytree_node=False, default=55.0)

    # Budget di normalizzazione eterosinaptica PRESINAPTICA: quanto peso totale
    # puo' distribuire in uscita un singolo sensore verso tutti i post-sinaptici
    # insieme (competizione per una risorsa presinaptica finita, complementare
    # a max_synaptic_budget). Un valore molto grande (default) = no-op.
    max_presynaptic_capacity: float = struct.field(pytree_node=False, default=1e6)

    # Scala temporale dello scaling omeostatico (vedi Synapse.py synapse_step).
    # 1.0 (default) = correzione istantanea e completa a ogni ciclo, identico al
    # comportamento originale. Valori piu' alti rendono il riequilibrio graduale
    # su molti cicli, come il vero synaptic scaling biologico (lento, non istantaneo).
    homeostatic_tau: float = struct.field(pytree_node=False, default=1.0)

    # Regola Delta Locale (azione integrale su code, vedi Synapse.py synapse_step).
    # False (default) = comportamento originale, covarianza pura. Attiva solo per
    # le popolazioni che ricevono un target_signal esterno (es. i motori nel
    # readout supervisionato).
    use_delta_rule: bool = struct.field(pytree_node=False, default=False)
    delta_rule_ki_ts: float = struct.field(pytree_node=False, default=1.0)
    delta_rule_q_capacity: float = struct.field(pytree_node=False, default=10.0)

    # Stagionatura per-sinapsi (vedi Synapse.py synapse_step): dopo quanti cicli
    # senza un evento di rinforzo/punizione su quella connessione si scarica un
    # pacchetto fisso di peso (decay_quantum). Confronto contro una soglia intera,
    # non una formula. decay_quantum=0.0 (default) = nessuna stagionatura, nessun
    # cambiamento per chi non la configura.
    delta_rule_decay_period: int = struct.field(pytree_node=False, default=50)
    delta_rule_decay_quantum: float = struct.field(pytree_node=False, default=0.0)

    # Attenzione top-down: un neurone dominante (vedi dominance_hold) inietta
    # corrente eccitatoria all'indietro sui presinaptici da cui dipende di piu'
    # (stesso filo del peso imparato, percorso al contrario), rinforzando la
    # propria evidenza durante la stessa presentazione. 0.0 (default) = nessun
    # feedback, comportamento originale invariato.
    attention_gain: float = struct.field(pytree_node=False, default=0.0)

def create_topology(n_neurons: int, n_branches: int, max_synaptic_budget: float = 55.0, homeostatic_tau: float = 1.0,
                     use_delta_rule: bool = False, delta_rule_ki_ts: float = 1.0, delta_rule_q_capacity: float = 10.0,
                     delta_rule_decay_period: int = 50, delta_rule_decay_quantum: float = 0.0,
                     max_presynaptic_capacity: float = 1e6, attention_gain: float = 0.0) -> NetworkTopology:
    return NetworkTopology(
        n_neurons=n_neurons,
        n_branches=n_branches,
        w_exc=jnp.zeros((n_neurons, n_neurons), dtype=jnp.float32),
        w_inh=jnp.zeros((n_neurons, n_neurons), dtype=jnp.float32),
        w_shunt=jnp.zeros((n_neurons, n_neurons), dtype=jnp.float32),
        axon_delays=jnp.ones((n_neurons, n_neurons), dtype=jnp.int32),
        branch_map=jnp.zeros((n_neurons, n_neurons), dtype=jnp.int32),
        plasticity_mask=jnp.zeros((n_neurons, n_neurons), dtype=jnp.bool_),
        dominance_hold=jnp.ones(n_neurons, dtype=jnp.int32),
        max_synaptic_budget=max_synaptic_budget,
        max_presynaptic_capacity=max_presynaptic_capacity,
        homeostatic_tau=homeostatic_tau,
        use_delta_rule=use_delta_rule,
        delta_rule_ki_ts=delta_rule_ki_ts,
        delta_rule_q_capacity=delta_rule_q_capacity,
        delta_rule_decay_period=delta_rule_decay_period,
        delta_rule_decay_quantum=delta_rule_decay_quantum,
        attention_gain=attention_gain
    )
