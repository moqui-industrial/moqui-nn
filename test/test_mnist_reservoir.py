import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax
import jax.numpy as jnp
import numpy as np
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
    HID_LO, HID_HI = N_SENSORS, N_SENSORS + N_HIDDEN
    MOT_LO, MOT_HI = HID_HI, N

    BRANCHES = 1
    MAX_DELAY = 1

    # -------------------------------------------------------------
    # Impianto: pattern separation reale (analogo al mushroom body /
    # giro dentato) tramite competizione fisica, non un top-k esterno.
    # Ogni neurone hidden e' un ActuatorCore + coda di Little completi,
    # esattamente come sensori e motori: nessuna scorciatoia numerica.
    # -------------------------------------------------------------
    # Guadagno dell'inibizione laterale tra neuroni hidden: e' un setpoint
    # dell'impianto (interlock reale via sinapsi inibitoria), non una formula
    # ad-hoc. Va tarato come qualunque altro parametro fisico della rete.
    HIDDEN_LATERAL_INHIB = 50.0

    # Budget di normalizzazione eterosinaptica: il default (55.0) e' tarato
    # per il fan-in dei 64 sensori dell'esperimento diretto (test_mnist_snn).
    # Qui il fan-in plastico e' quello dei 500 neuroni hidden, quindi il
    # budget va riscalato in proporzione al numero atteso di presinaptici
    # attivi (altrimenti ogni sinapsi resta annacquata e i pesi collassano).
    SYNAPTIC_BUDGET = 600.0

    # Attenzione top-down: il motore attualmente dominante inietta corrente
    # eccitatoria all'indietro sui neuroni hidden da cui dipende di piu' (stesso
    # filo del peso gia' imparato, percorso al contrario), rinforzando la propria
    # evidenza per il resto della presentazione. Attiva solo in inferenza.
    ATTENTION_GAIN = 0.5

    # Stagionatura per-sinapsi: dopo quanti cicli senza un evento di rinforzo/
    # punizione su quella specifica connessione si scarica un pacchetto fisso di
    # peso. Intervallo delimitato empiricamente: 300->10/10 pareggiati, 100->9/10,
    # 30->5/10, 2->1/10 ma sbagliato (schiaccia anche il segnale corretto).
    # Torniamo al valore piu' mite (30): la specializzazione la fa ora soprattutto
    # il budget presinaptico qui sotto, non un decadimento aggressivo.
    DELTA_RULE_DECAY_PERIOD = 30
    DELTA_RULE_DECAY_QUANTUM = 1.0

    # Budget presinaptico: quanto peso totale puo' distribuire in uscita un
    # singolo sensore verso TUTTI i motori insieme. Misurato empiricamente su
    # un training parziale: con 25 la soglia non vincolava quasi nessuno
    # (p90=10.3, p99=15.4 dopo 300 immagini) - la restringo a 12.
    MAX_PRESYNAPTIC_CAPACITY = 12.0

    # Tempo di mantenimento del gettone di dominanza per i neuroni hidden: quanti
    # cicli resta valido il fatto "ho vinto", oltre il proprio refrattario. Con 1
    # (il default per sensori/motori, invariato) l'inibizione competitiva torna a
    # essere legata al singolo ciclo di spike e si ottiene il carosello a rotazione
    # gia' osservato. Un valore maggiore fa persistere il vincitore per gran parte
    # della presentazione, evitando che i concorrenti abbiano una finestra libera
    # ogni volta che il vincitore va in refrattario.
    HIDDEN_DOMINANCE_HOLD = 8

    # Impulso sensoriale transitorio (fronte di salita), non livello sostenuto:
    # il fotorecettore reale si adatta e smette di rispondere a una luce costante
    # entro poche decine di ms, e l'occhio non sta mai fermo abbastanza a lungo
    # da mantenere un ingresso continuo. Teniamo il pixel come corrente per pochi
    # cicli iniziali, poi lo spegniamo - la competizione ha cosi' una finestra
    # realistica e finita, invece di un tempo indefinito in cui chiunque, anche
    # debolmente eccitato, finisce per superare la soglia.
    PULSE_DURATION = 5

    # Convergenza sensori->hidden: il dato anatomico reale (Caron et al. 2013,
    # cellule di Kenyon nella Drosophila) e' che ogni cellula di espansione
    # riceve input da SOLO ~6-7 neuroni di proiezione su circa 50 disponibili
    # (~12-14%), non una densita' del 30%. In piu' una cellula di Kenyon non
    # scatta per un solo ingresso forte: serve la coincidenza di piu' dei suoi
    # pochi input attivi insieme. Replichiamo entrambi i fatti: pochi fili per
    # rilevatore, e un guadagno per filo abbastanza basso che nessuno da solo basti.
    K_CONVERGENCE = 6

    rng = np.random.default_rng(123)

    w_exc_np = np.zeros((N, N), dtype=np.float32)
    w_inh_np = np.zeros((N, N), dtype=np.float32)
    plasticity_np = np.zeros((N, N), dtype=bool)

    # -------------------------------------------------------------
    # 1. Connessioni Fisse: Sensori -> Interneuroni (Il Reservoir)
    # -------------------------------------------------------------
    # Ogni neurone hidden sceglie esattamente K_CONVERGENCE sensori casuali
    # (senza reinserimento) - basso fan-in fisso, come le cellule di Kenyon,
    # non una densita' di Bernoulli uniforme sull'intera popolazione.
    conn_mask = np.zeros((N_SENSORS, N_HIDDEN), dtype=bool)
    for h in range(N_HIDDEN):
        chosen = rng.choice(N_SENSORS, size=K_CONVERGENCE, replace=False)
        conn_mask[chosen, h] = True

    # Guadagno per sinapsi ridotto (rispetto alla versione precedente 0-35):
    # nessun singolo filo deve bastare da solo a superare la soglia del dendrite,
    # cosi' serve la coincidenza di piu' dei ~6 input attivi insieme.
    rand_exc = rng.uniform(0.0, 12.0, size=(N_SENSORS, N_HIDDEN)).astype(np.float32)
    rand_inh = rng.uniform(0.0, 4.0, size=(N_SENSORS, N_HIDDEN)).astype(np.float32)
    # 80% eccitatorio, 20% inibitorio (stessa proporzione dell'originale)
    parity = (np.add.outer(np.arange(N_SENSORS), np.arange(N_HIDDEN)) % 5) != 0
    w_exc_np[:N_SENSORS, HID_LO:HID_HI] = np.where(conn_mask & parity, rand_exc, 0.0)
    w_inh_np[:N_SENSORS, HID_LO:HID_HI] = np.where(conn_mask & ~parity, rand_inh, 0.0)

    # -------------------------------------------------------------
    # 2. Inibizione Laterale Hidden <-> Hidden (Pattern Separation)
    # -------------------------------------------------------------
    # Vera competizione fisica tra i neuroni hidden: chi spara prima/piu'
    # forte inibisce gli altri attraverso le stesse sinapsi inibitorie usate
    # altrove nella rete. Emerge cosi' un codice sparso stabile, invece della
    # attivita' diffusa e priva di struttura vista senza questo meccanismo.
    hidden_inhib = np.full((N_HIDDEN, N_HIDDEN), HIDDEN_LATERAL_INHIB, dtype=np.float32)
    np.fill_diagonal(hidden_inhib, 0.0)
    w_inh_np[HID_LO:HID_HI, HID_LO:HID_HI] = hidden_inhib

    # -------------------------------------------------------------
    # 3. Connessioni Plastiche: Interneuroni -> Motori
    # -------------------------------------------------------------
    plasticity_np[HID_LO:HID_HI, MOT_LO:MOT_HI] = True

    # -------------------------------------------------------------
    # 4. Inibizione Laterale Winner-Takes-All (Motori -> Motori)
    # -------------------------------------------------------------
    motor_inhib = np.full((N_MOTORS, N_MOTORS), 5000.0, dtype=np.float32)
    np.fill_diagonal(motor_inhib, 0.0)
    w_inh_np[MOT_LO:MOT_HI, MOT_LO:MOT_HI] = motor_inhib

    # Anche i motori vanno in carosello se il vincitore non resta dominante oltre
    # il proprio refrattario: con pesi appresi solo moderatamente differenziati
    # (non un caso estremo), il margine del vincitore reale non basta a escludere
    # gli altri prima che tornino in gioco - e con 10 motori a ~4 cicli ciascuno
    # (spike+refrattario), 40 cicli di test bastano esattamente per farli sparare
    # tutti in sequenza. Stesso gettone di dominanza gia' usato per i neuroni hidden.
    MOTOR_DOMINANCE_HOLD = 30

    dominance_hold_np = np.ones(N, dtype=np.int32)
    dominance_hold_np[HID_LO:HID_HI] = HIDDEN_DOMINANCE_HOLD
    dominance_hold_np[MOT_LO:MOT_HI] = MOTOR_DOMINANCE_HOLD

    topo = create_topology(N, BRANCHES, max_synaptic_budget=SYNAPTIC_BUDGET, use_delta_rule=True,
                            delta_rule_decay_period=DELTA_RULE_DECAY_PERIOD,
                            delta_rule_decay_quantum=DELTA_RULE_DECAY_QUANTUM,
                            max_presynaptic_capacity=MAX_PRESYNAPTIC_CAPACITY, attention_gain=ATTENTION_GAIN)
    topo = topo.replace(
        w_exc=jnp.array(w_exc_np),
        w_inh=jnp.array(w_inh_np),
        plasticity_mask=jnp.array(plasticity_np),
        dominance_hold=jnp.array(dominance_hold_np)
    )
    plasticity_mask = topo.plasticity_mask

    state = init_network_state(topo, max_delay=MAX_DELAY, max_log_size=10)
    SCANS_PER_IMAGE = 20

    # ---------------------------------------------------------
    # TRAINING (Online Teacher Forcing)
    # ---------------------------------------------------------
    print(f"Training su {len(X_train)} immagini con Reservoir Computing (500 Interneuroni, "
          f"inibizione laterale {HIDDEN_LATERAL_INHIB}, budget sinaptico {SYNAPTIC_BUDGET})...")
    start_time = time.time()

    bb_train = Blackboard(dopamine_level=10.0, serotonin_level=0.0, sleep_mode=False)
    run_jitted = jax.jit(run_network_scan)

    EPOCHS = 2
    for epoch in range(EPOCHS):
        hidden_active_total = 0
        for idx in range(len(X_train)):
            state = reset_neuron_dynamics(state)
            pixels = X_train[idx]
            label = y_train[idx]

            ext_exc = jnp.zeros((SCANS_PER_IMAGE, N))
            ext_exc = ext_exc.at[:PULSE_DURATION, :N_SENSORS].set(pixels * 300.0)

            motor_idx = MOT_LO + label
            ext_exc = ext_exc.at[:, motor_idx].set(800.0)

            # Nessuna soppressione esterna sui motori non-bersaglio: se un motore
            # sbagliato spara comunque (perche' condivide sensori ambigui col vero
            # bersaglio), quello e' il segnale utile - un errore negativo esplicito
            # sulle sue sinapsi attive in quel momento (depressione eterosinaptica),
            # non silenzio. Il bersaglio vero vince comunque grazie alla spinta
            # esterna (800.0 sopra) piu' l'inibizione laterale interna tra motori.
            ext_inh = jnp.zeros((SCANS_PER_IMAGE, N))

            disable = jnp.zeros((SCANS_PER_IMAGE, N), dtype=jnp.bool_)

            # Bersaglio per la regola delta locale: "questo motore dovrebbe essere
            # attivo ora" - lo stesso segnale del teacher forcing, fornito dal
            # chiamante (qui), non calcolato dentro Synapse.py.
            ext_target = jnp.zeros((SCANS_PER_IMAGE, N), dtype=jnp.bool_)
            ext_target = ext_target.at[:, motor_idx].set(True)

            state, spike_history = run_jitted(state, topo, bb_train, ext_exc, ext_inh, disable, ext_target)
            hidden_active_total += int(jnp.sum(spike_history[:, HID_LO:HID_HI]))

        current_max_w = jnp.max(jnp.where(plasticity_mask, state.synapses.weight, 0.0))
        avg_hidden_spikes = hidden_active_total / len(X_train)
        print(f"Epoca {epoch+1}/{EPOCHS} completata. Max learned weight: {current_max_w:.2f} | "
              f"Spike hidden medi/immagine: {avg_hidden_spikes:.1f} "
              f"({100*avg_hidden_spikes/(N_HIDDEN*SCANS_PER_IMAGE):.1f}% duty cycle)")

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
        ext_exc = ext_exc.at[:PULSE_DURATION, :N_SENSORS].set(pixels * 300.0)

        ext_inh = jnp.zeros((TEST_SCANS, N))
        disable = jnp.zeros((TEST_SCANS, N), dtype=jnp.bool_)

        state_test, spike_history = run_jitted(state_test, topo, bb_test, ext_exc, ext_inh, disable)

        hidden_spikes = jnp.sum(spike_history[:, HID_LO:HID_HI])
        motor_spikes = jnp.sum(spike_history[:, MOT_LO:MOT_HI], axis=0)  # Shape [10]

        if idx == 0:
            print(f"Esempio Test 0 Hidden Spikes totali: {hidden_spikes}")
            print(f"Esempio Test 0 Motor Spikes: {motor_spikes}")

        prediction = jnp.argmax(motor_spikes)
        if jnp.max(motor_spikes) == 0:
            pass  # Rete incerta
        elif prediction == label:
            correct += 1

    test_time = time.time() - start_time
    test_acc = correct / len(X_test)

    print(f"--- Risultati SNN Biologica (Reservoir) ---")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(f"Inferenza completata in {test_time:.2f} s")

if __name__ == "__main__":
    test_mnist_reservoir()
