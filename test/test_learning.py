import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False

from moqui_nn.NNFacade import create_topology
from moqui_nn.DeviceManager import init_network_state, run_network_scan, replay_log
from moqui_nn.Main import Blackboard

def test_classical_conditioning():
    print("--- Pavlovian Conditioning Test ---")
    N = 3
    BRANCHES = 1
    MAX_DELAY = 1
    LOG_SIZE = 200
    
    topo = create_topology(N, BRANCHES)
    
    # 0 = Odor A (Neutro)
    # 1 = Dolore (Incondizionato)
    # 2 = Motore (Fuga)
    
    # Wiring innato (Riflesso incondizionato)
    topo = topo.replace(w_exc=topo.w_exc.at[1, 2].set(20.0))
    
    # Connessione Odore -> Motore esiste ma è a 0. Abilitiamo plasticità
    topo = topo.replace(w_exc=topo.w_exc.at[0, 2].set(0.0))
    new_mask = topo.plasticity_mask.at[0, 2].set(True)
    topo = topo.replace(plasticity_mask=new_mask)
    
    # Init
    state = init_network_state(topo, MAX_DELAY, LOG_SIZE)
    # Impostiamo dopamina a 0.0 durante l'awake per disabilitare l'apprendimento online!
    # Questo forzerà l'apprendimento ad avvenire SOLO tramite il LogRingBuffer durante lo sleep.
    bb_awake = Blackboard(dopamine_level=0.0, serotonin_level=0.0, sleep_mode=False)
    
    # ---------------------------------------------------------
    # FASE 1: Awake (Stimolo Neutro + Stimolo Incondizionato)
    # ---------------------------------------------------------
    print("Fase 1: Awake (Associazione Odore + Dolore)")
    M_awake = 100
    ext_exc = jnp.zeros((M_awake, N))
    ext_inh = jnp.zeros((M_awake, N))
    disable = jnp.zeros((M_awake, N), dtype=jnp.bool_)
    
    # Accendiamo Odore e Dolore a tratti brevi (es. scan 10-15 e 30-35)
    ext_exc = ext_exc.at[10:20, 0].set(400.0) # Odore
    ext_exc = ext_exc.at[10:20, 1].set(400.0) # Dolore
    ext_exc = ext_exc.at[40:50, 0].set(400.0)
    ext_exc = ext_exc.at[40:50, 1].set(400.0)
    ext_exc = ext_exc.at[70:80, 0].set(400.0)
    ext_exc = ext_exc.at[70:80, 1].set(400.0)
    
    run_jitted = jax.jit(run_network_scan)
    state, spike_history_awake = run_jitted(state, topo, bb_awake, ext_exc, ext_inh, disable)
    
    print(f"  AWAKE Spikes Odore (0): {jnp.where(spike_history_awake[:, 0])[0]}")
    print(f"  AWAKE Spikes Dolore (1): {jnp.where(spike_history_awake[:, 1])[0]}")
    print(f"  AWAKE Spikes Motore (2): {jnp.where(spike_history_awake[:, 2])[0]}")
    
    print(f"  Peso Odore -> Motore DOPO l'awake (nessun apprendimento online per via del delay STP): {state.synapses.weight[0, 2]:.2f}")
    
    # ---------------------------------------------------------
    # FASE 2: Sleep (Replay offline)
    # ---------------------------------------------------------
    print("Fase 2: Sleep (Log Replay & Consolidamento)")
    # Dopamina alta per imparare le memorie emotivamente cariche
    bb_sleep = Blackboard(dopamine_level=10.0, serotonin_level=0.0, sleep_mode=True)
    
    replay_jitted = jax.jit(replay_log)
    # Facciamo vari loop di replay (es. 200 epoche di sonno)
    for epoch in range(200):
        state = replay_jitted(state, topo, bb_sleep)
        
    print(f"  Peso Odore -> Motore DOPO il Replay (Offline): {state.synapses.weight[0, 2]:.2f}")
    
    # ---------------------------------------------------------
    # FASE 3: Test (Solo Stimolo Neutro)
    # ---------------------------------------------------------
    print("Fase 3: Test (Solo Odore)")
    # Riscriviamo la topologia con i pesi appresi!
    topo = topo.replace(w_exc=state.synapses.weight)
    
    M_test = 50
    ext_exc_test = jnp.zeros((M_test, N))
    ext_inh_test = jnp.zeros((M_test, N))
    disable_test = jnp.zeros((M_test, N), dtype=jnp.bool_)
    ext_exc_test = ext_exc_test.at[10:20, 0].set(400.0) # SOLO ODORE!
    
    # Nuovo stato vuoto per il test
    state_test = init_network_state(topo, MAX_DELAY, LOG_SIZE)
    state_test, spike_history_test = run_jitted(state_test, topo, bb_awake, ext_exc_test, ext_inh_test, disable_test)
    
    print(f"  Spikes Odore (0) timing: {jnp.where(spike_history_test[:, 0])[0]}")
    print(f"  Spikes Dolore (1) timing: {jnp.where(spike_history_test[:, 1])[0]}")
    print(f"  Spikes Motore (2) timing: {jnp.where(spike_history_test[:, 2])[0]}")
    
    spikes_m_test = spike_history_test[:, 2]
    print(f"  Il Motore (Fuga) ha sparato? {'SI' if jnp.any(spikes_m_test) else 'NO'}")
    
    if HAVE_PLT:
        os.makedirs("./outputs", exist_ok=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.eventplot(jnp.where(spike_history_test[:, 0])[0], lineoffsets=2, colors='blue', label="Odor Sensor")
        ax.eventplot(jnp.where(spike_history_test[:, 1])[0], lineoffsets=1, colors='red', label="Pain Sensor")
        ax.eventplot(jnp.where(spike_history_test[:, 2])[0], lineoffsets=0, colors='black', label="Motor (Escape)")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["Motor", "Pain", "Odor"])
        ax.set_title("Test (Only Odor is present) - Post Conditioning")
        fig.tight_layout()
        fig.savefig("./outputs/learning_test.png")
        print("  -> Grafico salvato in ./outputs/learning_test.png")

if __name__ == "__main__":
    test_classical_conditioning()
