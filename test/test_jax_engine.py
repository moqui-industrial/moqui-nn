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

from moqui_nn.Constants import *
from moqui_nn.Neuron import init_neuron_state, neuron_step
from moqui_nn.NNFacade import create_topology
from moqui_nn.DeviceManager import init_network_state, run_network_scan
from moqui_nn.Main import Blackboard

def test_jax_oscillator():
    print("--- JAX Vectorized Engine Test: E-I Oscillator ---")
    
    # 2 Neurons: 0 is Exc, 1 is Inh
    N = 2
    BRANCHES = 1
    MAX_DELAY = 4 # Dead time (4 scans)
    
    topo = create_topology(N, BRANCHES)
    
    # Wiring
    # E(0) eccita I(1)
    topo = topo.replace(w_exc=topo.w_exc.at[0, 1].set(1500.0))
    
    # I(1) inibisce E(0)
    topo = topo.replace(w_inh=topo.w_inh.at[1, 0].set(1500.0))
    
    # Init state
    state = init_network_state(topo, MAX_DELAY)
    
    # Sequences (100 scans)
    M = 100
    ext_exc = jnp.zeros((M, N))
    ext_inh = jnp.zeros((M, N))
    disable = jnp.zeros((M, N), dtype=jnp.bool_)
    
    # Drive costante su E
    ext_exc = ext_exc.at[:, 0].set(400.0)
    
    # Blackboard
    bb = Blackboard(dopamine_level=1.0, serotonin_level=0.0, sleep_mode=False)
    
    print("  Compiling and running lax.scan...")
    # JIT compilazione del loop di rete
    run_jitted = jax.jit(run_network_scan)
    final_state, spike_history = run_jitted(state, topo, bb, ext_exc, ext_inh, disable)
    
    print("  Esecuzione completata.")
    
    # spike_history ha shape [M, N]
    spikes_e = spike_history[:, 0]
    spikes_i = spike_history[:, 1]
    
    print(f"  Spikes E: {jnp.where(spikes_e)[0].tolist()}")
    print(f"  Spikes I: {jnp.where(spikes_i)[0].tolist()}")
    
    if HAVE_PLT:
        os.makedirs("./outputs", exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.eventplot(jnp.where(spikes_e)[0], lineoffsets=1, colors='r', label="E (Exc)")
        ax.eventplot(jnp.where(spikes_i)[0], lineoffsets=0, colors='b', label="I (Inh)")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["I", "E"])
        ax.set_xlim(0, M)
        ax.set_title("JAX Vectorized E-I Oscillator")
        fig.tight_layout()
        fig.savefig("./outputs/jax_oscillator.png", dpi=130)
        print("  -> Grafico salvato in ./outputs/jax_oscillator.png")
    else:
        print("  -> Matplotlib non disponibile, grafico saltato.")

if __name__ == "__main__":
    test_jax_oscillator()
