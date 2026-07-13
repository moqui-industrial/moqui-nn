import jax
import jax.numpy as jnp
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

from moqui_nn.NNFacade import create_topology
from moqui_nn.DeviceManager import init_network_state, run_network_scan
from moqui_nn.Main import Blackboard

def reset_neuron_dynamics(state):
    neurons = state.neurons
    dendrite = neurons.dendrite.replace(
        accumulator=jnp.zeros_like(neurons.dendrite.accumulator),
        fatigue=jnp.zeros_like(neurons.dendrite.fatigue)
    )
    actuator = neurons.actuator.replace(
        status=jnp.zeros_like(neurons.actuator.status),
        spike_timer=jnp.zeros_like(neurons.actuator.spike_timer),
        refractory_timer=jnp.zeros_like(neurons.actuator.refractory_timer),
        enable=jnp.zeros_like(neurons.actuator.enable, dtype=jnp.bool_)
    )
    return state.replace(
        neurons=neurons.replace(dendrite=dendrite, actuator=actuator),
        spike_buffer=jnp.zeros_like(state.spike_buffer),
        buffer_idx=jnp.int32(0),
        axon_refractory_timer=jnp.zeros_like(state.axon_refractory_timer),
        dopamine_buffer=jnp.zeros_like(state.dopamine_buffer),
        dopamine_idx=jnp.int32(0),
        dominance_timer=jnp.zeros_like(state.dominance_timer)
    )

def test_mnist_snn():
    print("--- Biological SNN (PLC-style) su Scikit-Learn Digits ---")
    digits = load_digits()
    X = digits.data / 16.0 
    y = digits.target
    
    # Per velocità della SNN in JAX locale, prendiamo un sottoinsieme
    # (es. 500 immagini di train e 100 di test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_test = X_test[:200]
    y_test = y_test[:200]
    
    POPULATION_SIZE = 5
    N_SENSORS = 64
    N_MOTORS = 10 * POPULATION_SIZE
    N = N_SENSORS + N_MOTORS
    BRANCHES = 1
    MAX_DELAY = 1
    
    topo = create_topology(N, BRANCHES)
    
    # Abilitiamo plasticità da tutti i sensori a tutti i motori
    # e inizializziamo i pesi a 0
    w_init = jnp.zeros((N, N), dtype=jnp.float32)
    
    # Maschera: Solo da Sensori a Motori
    mask = jnp.zeros((N, N), dtype=jnp.bool_)
    mask = mask.at[:N_SENSORS, N_SENSORS:].set(True)
    
    topo = topo.replace(plasticity_mask=mask)
    
    # Impostiamo i pesi iniziali solo dove la maschera è True
    w_exc = jnp.where(mask, w_init, 0.0)
    
    # Inibizione Laterale (WTA) tra le Popolazioni di Motori (senza inibizione intra-popolazione)
    w_inh = jnp.zeros((N, N), dtype=jnp.float32)
    for i in range(N_SENSORS, N):
        pop_i = (i - N_SENSORS) // POPULATION_SIZE
        for j in range(N_SENSORS, N):
            pop_j = (j - N_SENSORS) // POPULATION_SIZE
            if pop_i != pop_j:
                w_inh = w_inh.at[i, j].set(5000.0)
                
    topo = topo.replace(w_exc=w_exc, w_inh=w_inh)
    
    state = init_network_state(topo, max_delay=MAX_DELAY, max_log_size=10)
    
    SCANS_PER_IMAGE = 40
    
    # ---------------------------------------------------------
    # TRAINING (Online Teacher Forcing)
    # ---------------------------------------------------------
    print(f"Training su {len(X_train)} immagini con Teacher Forcing Hebbiano...")
    start_time = time.time()
    
    bb_train = Blackboard(dopamine_level=10.0, serotonin_level=0.0, sleep_mode=False)
    run_jitted = jax.jit(run_network_scan)
    
    for epoch in range(2): # 2 epoche per un consolidamento graduale dei pesi Hebbiani
        for idx in range(len(X_train)):
            # Resettiamo solo la fatica, l'accumulatore e lo stato dell'attuatore dei neuroni
            state = reset_neuron_dynamics(state)
            
            pixels = X_train[idx]
            label = y_train[idx]
            
            ext_exc = jnp.zeros((SCANS_PER_IMAGE, N))
            
            # Sensori: Rate coding continuo
            # Un pixel a 1.0 genera 300 di corrente eccitatoria (ci vorranno ~4 scan per sparare)
            ext_exc = ext_exc.at[:, :N_SENSORS].set(pixels * 300.0)
            
            # Popolazione Motore Target: Teacher Forcing
            target_indices = [N_SENSORS + label * POPULATION_SIZE + i for i in range(POPULATION_SIZE)]
            target_indices_arr = jnp.array(target_indices)
            ext_exc = ext_exc.at[:, target_indices_arr].set(800.0)
            
            ext_inh = jnp.zeros((SCANS_PER_IMAGE, N))
            # Inibizione massiccia su tutti i motori
            ext_inh = ext_inh.at[:, N_SENSORS:].set(800.0)
            # Rimuoviamo inibizione solo per la popolazione target
            ext_inh = ext_inh.at[:, target_indices_arr].set(0.0)
            
            disable = jnp.zeros((SCANS_PER_IMAGE, N), dtype=jnp.bool_)
            
            # Scan
            state, _ = run_jitted(state, topo, bb_train, ext_exc, ext_inh, disable)
            
    train_time = time.time() - start_time
    print(f"Training Completato in {train_time:.2f} s")
    
    # Fissiamo i pesi (Dopamina a 0)
    # I pesi sono normalizzati ed equilibrati tramite la normalizzazione eterosinaptica, li usiamo direttamente
    topo = topo.replace(w_exc=state.synapses.weight)
    
    print(f"Max weight in topology: {jnp.max(topo.w_exc)}")
    print(f"Min weight in topology (active): {jnp.min(jnp.where(mask, topo.w_exc, 999.0))}")
    print(f"Weights from Sensor 36 to all Motors: {topo.w_exc[36, N_SENSORS:]}")
    
    # ---------------------------------------------------------
    # TESTING
    # ---------------------------------------------------------
    print(f"Inferenza su {len(X_test)} immagini...")
    bb_test = Blackboard(dopamine_level=0.0, serotonin_level=0.0, sleep_mode=False)
    
    correct = 0
    start_time = time.time()
    
    TEST_SCANS = 40
    for idx in range(len(X_test)):
        # Resettiamo lo stato per pulire gli accumulatori ed evitare accumulo di fatica tra immagini
        state_test = init_network_state(topo, MAX_DELAY, max_log_size=10)
        pixels = X_test[idx]
        label = y_test[idx]
        
        ext_exc = jnp.zeros((TEST_SCANS, N))
        ext_exc = ext_exc.at[:, :N_SENSORS].set(pixels * 300.0)
        
        ext_inh = jnp.zeros((TEST_SCANS, N))
        disable = jnp.zeros((TEST_SCANS, N), dtype=jnp.bool_)
        
        state_test, spike_history = run_jitted(state_test, topo, bb_test, ext_exc, ext_inh, disable)
        
        if idx == 0:
            print(f"Esempio Test 0 ext_exc su Motori: {jnp.sum(ext_exc[:, N_SENSORS:])}")
        
        # Guardiamo gli spike dei Motori
        motor_spikes = jnp.sum(spike_history[:, N_SENSORS:], axis=0) # Shape [50]
        if idx == 0:
            print(f"Esempio Test 0 Motor Spikes: {motor_spikes}")
            print(f"Esempio Test 0 Spikes Popolazione 0: {motor_spikes[:POPULATION_SIZE]}")
        
        # La predizione è la popolazione con più spike cumulativi (voto di maggioranza)
        pop_spikes = motor_spikes.reshape((10, POPULATION_SIZE))
        votes = jnp.sum(pop_spikes, axis=1) # Shape [10]
        prediction = jnp.argmax(votes)
        
        if jnp.max(votes) == 0:
            # Nessuno spike (rete incerta)
            pass
        elif prediction == label:
            correct += 1
            
    test_time = time.time() - start_time
    test_acc = correct / len(X_test)
    
    print(f"--- Risultati SNN Biologica ---")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Inferenza completata in {test_time:.2f} s")

if __name__ == "__main__":
    test_mnist_snn()
