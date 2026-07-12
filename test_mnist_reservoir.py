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
        dopamine_idx=jnp.int32(0)
    )

def test_mnist_reservoir():
    print("--- Biological SNN (Reservoir Computing) su Scikit-Learn Digits ---")
    digits = load_digits()
    X = digits.data / 16.0 
    y = digits.target
    
    # Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = X_train[:1000]
    y_train = y_train[:1000]
    X_test = X_test[:200]
    y_test = y_test[:200]
    
    N_SENSORS = 64
    N_HIDDEN = 500
    N_MOTORS = 10
    N = N_SENSORS + N_HIDDEN + N_MOTORS
    
    BRANCHES = 1
    MAX_DELAY = 1
    
    topo = create_topology(N, BRANCHES)
    
    # Generatore casuale
    rng = jax.random.PRNGKey(123)
    rng_exc, rng_inh, rng_mask = jax.random.split(rng, 3)
    
    # -------------------------------------------------------------
    # 1. Connessioni Fisse: Sensori -> Interneuroni (Il Reservoir)
    # -------------------------------------------------------------
    # Creiamo connessioni sparse e casuali (eccitatorie e inibitorie)
    # per proiettare lo spazio da 64D a 500D.
    # Diamo più peso all'eccitazione per permettere al reservoir di attivarsi!
    prob_connection = 0.3 # 30% di densità
    
    rand_exc = jax.random.uniform(rng_exc, (N, N), minval=0.0, maxval=35.0)
    rand_inh = jax.random.uniform(rng_inh, (N, N), minval=0.0, maxval=10.0)
    conn_mask = jax.random.uniform(rng_mask, (N, N)) < prob_connection
    
    w_exc_initial = jnp.zeros((N, N))
    w_inh_initial = jnp.zeros((N, N))
    
    # Applichiamo i pesi casuali fissi solo da Sensori a Hidden
    for s in range(N_SENSORS):
        for h in range(N_SENSORS, N_SENSORS + N_HIDDEN):
            if conn_mask[s, h]:
                # 80% eccitatorio, 20% inibitorio
                if (s + h) % 5 != 0:
                    w_exc_initial = w_exc_initial.at[s, h].set(rand_exc[s, h])
                else:
                    w_inh_initial = w_inh_initial.at[s, h].set(rand_inh[s, h])
                    
    # -------------------------------------------------------------
    # 2. Connessioni Plastiche: Interneuroni -> Motori
    # -------------------------------------------------------------
    # Inizializziamo a 0. L'apprendimento creerà le connessioni.
    plasticity_mask = jnp.zeros((N, N), dtype=jnp.bool_)
    
    # La plasticità è attiva solo dai neuroni Hidden ai Motori
    for h in range(N_SENSORS, N_SENSORS + N_HIDDEN):
        for m in range(N_SENSORS + N_HIDDEN, N):
            plasticity_mask = plasticity_mask.at[h, m].set(True)
            
    # -------------------------------------------------------------
    # 3. Inibizione Laterale Winner-Takes-All (Motori -> Motori)
    # -------------------------------------------------------------
    for m1 in range(N_SENSORS + N_HIDDEN, N):
        for m2 in range(N_SENSORS + N_HIDDEN, N):
            if m1 != m2:
                w_inh_initial = w_inh_initial.at[m1, m2].set(5000.0)
                
    # Aggiorniamo la topologia
    topo = topo.replace(
        w_exc=w_exc_initial,
        w_inh=w_inh_initial,
        plasticity_mask=plasticity_mask
    )
    
    state = init_network_state(topo, max_delay=MAX_DELAY, max_log_size=10)
    SCANS_PER_IMAGE = 20
    
    # ---------------------------------------------------------
    # TRAINING (Online Teacher Forcing)
    # ---------------------------------------------------------
    print(f"Training su {len(X_train)} immagini con Reservoir Computing (500 Interneuroni)...")
    start_time = time.time()
    
    bb_train = Blackboard(dopamine_level=10.0, serotonin_level=0.0, sleep_mode=False)
    run_jitted = jax.jit(run_network_scan)
    
    # 10 epoche con BCM per testare l'equilibrio dinamico
    EPOCHS = 2
    for epoch in range(EPOCHS):
        for idx in range(len(X_train)):
            state = reset_neuron_dynamics(state)
            pixels = X_train[idx]
            label = y_train[idx]
            
            ext_exc = jnp.zeros((SCANS_PER_IMAGE, N))
            
            # Sensori: Rate coding
            ext_exc = ext_exc.at[:, :N_SENSORS].set(pixels * 300.0)
            
            # Motore: Teacher Forcing (forziamo lo spike del motore corretto)
            motor_idx = N_SENSORS + N_HIDDEN + label
            ext_exc = ext_exc.at[:, motor_idx].set(800.0)
            
            ext_inh = jnp.zeros((SCANS_PER_IMAGE, N))
            # Inibizione massiccia sui motori non-target per prevenire spike spuri e falsi positivi
            ext_inh = ext_inh.at[:, N_SENSORS + N_HIDDEN:].set(800.0)
            ext_inh = ext_inh.at[:, motor_idx].set(0.0)
            
            disable = jnp.zeros((SCANS_PER_IMAGE, N), dtype=jnp.bool_)
            
            state, _ = run_jitted(state, topo, bb_train, ext_exc, ext_inh, disable)
        
        # Stampa l'evoluzione a ogni epoca
        current_max_w = jnp.max(jnp.where(plasticity_mask, state.synapses.weight, 0.0))
        print(f"Epoca {epoch+1}/{EPOCHS} completata. Max learned weight: {current_max_w:.2f}")
            
    train_time = time.time() - start_time
    print(f"Training Completato in {train_time:.2f} s")
    
    final_w_exc = jnp.where(plasticity_mask, state.synapses.weight, topo.w_exc)
    topo = topo.replace(w_exc=final_w_exc)
    
    print(f"Max learned weight: {jnp.max(jnp.where(plasticity_mask, topo.w_exc, 0.0))}")
    
    # ---------------------------------------------------------
    # TESTING
    # ---------------------------------------------------------
    print(f"Inferenza su {len(X_test)} immagini...")
    bb_test = Blackboard(dopamine_level=0.0, serotonin_level=0.0, sleep_mode=False)
    
    correct = 0
    start_time = time.time()
    
    TEST_SCANS = 40
    for idx in range(len(X_test)):
        state_test = init_network_state(topo, MAX_DELAY, max_log_size=10)
        pixels = X_test[idx]
        label = y_test[idx]
        
        ext_exc = jnp.zeros((TEST_SCANS, N))
        ext_exc = ext_exc.at[:, :N_SENSORS].set(pixels * 300.0)
        
        ext_inh = jnp.zeros((TEST_SCANS, N))
        disable = jnp.zeros((TEST_SCANS, N), dtype=jnp.bool_)
        
        state_test, spike_history = run_jitted(state_test, topo, bb_test, ext_exc, ext_inh, disable)
        
        # Guardiamo gli spike dei Motori e del Reservoir
        hidden_spikes = jnp.sum(spike_history[:, N_SENSORS:N_SENSORS + N_HIDDEN])
        motor_spikes = jnp.sum(spike_history[:, N_SENSORS + N_HIDDEN:], axis=0) # Shape [10]
        
        if idx == 0:
            print(f"Esempio Test 0 Hidden Spikes totali: {hidden_spikes}")
            print(f"Esempio Test 0 Motor Spikes: {motor_spikes}")
        
        # La predizione è il motore con più spike
        prediction = jnp.argmax(motor_spikes)
        if jnp.max(motor_spikes) == 0:
            pass # Rete incerta
        elif prediction == label:
            correct += 1
            
    test_time = time.time() - start_time
    test_acc = correct / len(X_test)
    
    print(f"--- Risultati SNN Biologica (Reservoir) ---")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Inferenza completata in {test_time:.2f} s")

if __name__ == "__main__":
    test_mnist_reservoir()
