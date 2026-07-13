#!/usr/bin/env python3
"""
industrial_snn_tests.py

Test per verificare i meccanismi fondamentali della nuova SNN
basata su pattern PLC:
- Test 1: Debito inibitorio (FLOOR)
- Test 2: Oscillatore E-I (Clock emergente dal Dead Time)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False

from neuron_industrial import (
    Neuron,
    Synapse,
    LogRingBuffer,
    LogEvent,
    TX_EXC,
    TX_INH,
    TX_SHUNT,
    DRIVER,
    Blackboard
)

def test_debito_inibitorio():
    print("--- Test 1: Debito inibitorio (FLOOR) ---")
    # Neurone con 2 rami (0: eccitatorio, 1: inibitorio)
    n = Neuron("n", n_masters=2)
    
    # Configuriamo le soglie per il test
    n.dendrite.THRESHOLD = 1000.0
    n.dendrite.CEILING = 1700.0
    n.dendrite.FLOOR = -700.0
    
    SCANS = 60
    acc_trace = []
    spikes = []
    
    inh_window = range(20, 24)
    
    for c in range(SCANS):
        n.reset_inputs()
        
        # Drive costante
        if c >= 2:
            n.dendrite.deposit(master_bit=0, charge=300.0, tag=TX_EXC | DRIVER)
            
        # Finestra inibitoria (forte colpo)
        if c in inh_window:
            n.dendrite.deposit(master_bit=1, charge=500.0, tag=TX_INH | DRIVER)
            
        n.run()
        acc_trace.append(n.dendrite.accumulator)
        if n.soma.enable:
            spikes.append(c)
            
    print(f"  Spike del soma ai cicli: {spikes}")
    print(f"  Osserva il debito sotto zero durante l'inibizione.")
    
    if HAVE_PLT:
        # Grafico
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(acc_trace, color="#1f77b4", lw=2, label="Occupazione Coda")
        ax.axhline(1000, color="k", ls="--", lw=1, label="THRESHOLD")
        ax.axhline(0, color="grey", lw=0.8)
        ax.axhline(-700, color="#d62728", ls=":", lw=1, label="FLOOR")
        ax.axvspan(20, 23, color="#d62728", alpha=0.15, label="Inibizione")
        for s in spikes:
            ax.axvline(s, color="#2ca02c", ls=":", alpha=0.5)
        
        ax.set_title("Test 1: Debito inibitorio e risalita")
        ax.set_xlabel("Scan")
        ax.set_ylabel("Accumulatore")
        ax.legend(loc="lower right")
        fig.savefig("./outputs/test_debito.png", dpi=130)
        print("  -> Grafico salvato in ./outputs/test_debito.png\n")
    else:
        print("  -> Matplotlib non disponibile, grafico saltato.\n")


def test_ei_oscillator():
    print("--- Test 2: Oscillatore E-I (Clock emergente) ---")
    
    # Costruiamo un anello: E accende I, I spegne E.
    # Usiamo il dendrite e l'assone.
    
    # N.B. In questo test semplificato bypassiamo il SOMA (che spara 0/1) 
    # e analizziamo solo l'interazione continua tra le code di due dendriti (E e I)
    # per misurare l'oscillazione dei potenziali medi.
    
    from neuron_industrial import SignalMgmtMC, AxonDelay
    
    DEAD_TIME = 4
    eExc = SignalMgmtMC(n_masters=1, leak_rate=200.0)
    iInh = SignalMgmtMC(n_masters=1, leak_rate=120.0)
    delay = AxonDelay(DEAD_TIME)
    
    eExc.THRESHOLD = 1000.0
    iInh.THRESHOLD = 1000.0
    
    E_trace = []
    I_trace = []
    G_trace = []
    
    W_E_DRIVE = 350.0  # Input esterno costante
    W_E_TO_I = 250.0
    W_I_TO_E = 600.0
    
    e_act = 0.0
    i_act = 0.0
    
    for c in range(120):
        # 1. Inibizione ritardata arriva a E
        inh_to_e = delay.update(i_act)
        
        # 2. Depositiamo cariche
        eExc.deposit(0, W_E_DRIVE, TX_EXC)
        eExc.deposit(0, (inh_to_e / 1000.0) * W_I_TO_E, TX_INH)
        
        iInh.deposit(0, (e_act / 1000.0) * W_E_TO_I, TX_EXC)
        
        # 3. Esecuzione (solo code)
        eExc.run()
        iInh.run()
        
        e_act = max(0.0, eExc.accumulator)
        i_act = max(0.0, iInh.accumulator)
        
        gate_open = i_act < 350.0
        
        E_trace.append(e_act)
        I_trace.append(i_act)
        G_trace.append(gate_open)
        
        # Reset per non far sparare ed azzerare tutto (simuliamo un firing rate analogico)
        eExc.enabledSensor = False
        iInh.enabledSensor = False
        
    print("  Oscillazione generata. Analisi visiva nel grafico.")
    
    if HAVE_PLT:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        ax1.plot(E_trace, color="#d62728", label="Eccitazione E")
        ax1.plot(I_trace, color="#1f77b4", label="Inibizione I")
        ax1.axhline(350, color="gray", ls=":", lw=0.8, label="Soglia Finestra")
        ax1.set_title("Anello E-I: Oscillatore di rete")
        ax1.legend(loc="upper right")
        
        ax2.fill_between(range(len(G_trace)), 0, [1 if g else 0 for g in G_trace], step="mid", color="#2ca02c", alpha=0.5)
        ax2.set_title("Finestra di coincidenza aperta")
        ax2.set_xlabel("Scan")
        
        fig.tight_layout()
        import os
        os.makedirs("./outputs", exist_ok=True)
        fig.savefig("./outputs/test_oscillator.png", dpi=130)
        print("  -> Grafico salvato in ./outputs/test_oscillator.png\n")
    else:
        print("  -> Matplotlib non disponibile, grafico saltato.\n")


def test_covariance_learning():
    print("--- Test 3: Apprendimento per Covarianza (Camma Elettronica) ---")
    # Due neuroni: pre e post. E una sinapsi tra di loro.
    pre = Neuron("pre", n_masters=1, dead_time=1)
    post = Neuron("post", n_masters=1, dead_time=1)
    syn = Synapse(transmitter=TX_EXC, weight=1.5)
    
    SCANS = 900
    w_trace = []
    
    # 3 epoche: IN FASE, SFASATO, IN FASE + PREMIO
    for c in range(SCANS):
        pre.reset_inputs()
        post.reset_inputs()
        modulator = 0.0
        
        period = 14
        phase = c % period
        epoch = c // 300
        
        # Forzature (Sorgenti esterne)
        if epoch == 0:
            # IN FASE: pre e post sparano ravvicinati
            if phase == 0: pre.dendrite.deposit(0, 1500.0, TX_EXC | DRIVER)
            if phase == 2: post.dendrite.deposit(0, 1500.0, TX_EXC | DRIVER)
        elif epoch == 1:
            # SFASATO: in antifase
            if phase == 0: pre.dendrite.deposit(0, 1500.0, TX_EXC | DRIVER)
            if phase == 7: post.dendrite.deposit(0, 1500.0, TX_EXC | DRIVER)
        else:
            # IN FASE + PREMIO (Modulatore)
            if phase == 0: pre.dendrite.deposit(0, 1500.0, TX_EXC | DRIVER)
            if phase == 2: post.dendrite.deposit(0, 1500.0, TX_EXC | DRIVER)
            if phase == 5: modulator = +1.0 # Dopamina/Reward
            
        # Run Pre
        pre_out = pre.run()
        # Axon out of Pre enters Synapse
        ch = syn.charge(pre_out > 0)
        # Synapse feeds Post
        if ch > 0:
            post.dendrite.deposit(0, ch, syn.transmitter) # Segnale propagato
            
        # Run Post
        post.run()
        
        # Aggiornamento Sinapsi
        syn.update(pre_event=(pre_out > 0), post_spike=post.soma.enable, modulator=modulator)
        w_trace.append(syn.weight)

    print(f"  Epoca 0 (IN FASE):      peso {w_trace[0]:.2f} -> {w_trace[299]:.2f}")
    print(f"  Epoca 1 (SFASATO):      peso {w_trace[300]:.2f} -> {w_trace[599]:.2f}")
    print(f"  Epoca 2 (IN FASE+MOD):  peso {w_trace[600]:.2f} -> {w_trace[-1]:.2f}")
    
    if HAVE_PLT:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(w_trace, color="#1f77b4", lw=2)
        ax.axvspan(0, 300, color="#2ca02c", alpha=0.10)
        ax.axvspan(300, 600, color="#d62728", alpha=0.10)
        ax.axvspan(600, 900, color="#1f77b4", alpha=0.10)
        ax.text(150, 4.5, "IN FASE", ha="center")
        ax.text(450, 4.5, "SFASATO", ha="center")
        ax.text(750, 4.5, "IN FASE + MOD", ha="center")
        ax.set_title("Evoluzione Peso Sinaptico (Covarianza)")
        ax.set_ylim(0, 5.5)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig("./outputs/test_covariance.png", dpi=130)
        print("  -> Grafico salvato in ./outputs/test_covariance.png\n")
    else:
        print("  -> Matplotlib non disponibile, grafico saltato.\n")


def test_replay_consolidation():
    print("--- Test 4: Consolidamento Offline (Replay via LogRingBuffer) ---")
    
    # Costruiamo il Ring Buffer (l'ippocampo)
    LOG_MAX_SIZE = 12
    buf = LogRingBuffer(LOG_MAX_SIZE)
    
    # Corteccia (matrice dei pesi: quantifica quanto un simbolo predice il prossimo)
    N_SYM = 5 # Stati A(0), B(1), C(2), D(3), E(4)
    cortex_weights = [[0.0] * N_SYM for _ in range(N_SYM)]
    LR = 0.20 # Learning rate
    
    def log_experience(sequence):
        # Fase ONLINE: solo log, nessun aggiornamento pesi.
        for i in range(len(sequence) - 1):
            ev = LogEvent(sequence[i], sequence[i + 1])
            last = buf.last()
            if last is not None and last == ev:
                buf.increment_last() # Comprime le ripetizioni (repeatCount)
            else:
                buf.push(ev)
                
    def offline_replay(cycles):
        # Fase OFFLINE: rilegge il buffer e aggiorna la corteccia.
        # Usa il repeatCount come moltiplicatore di priorita'.
        for _ in range(cycles):
            for ev in buf.entries():
                strength = 1.0 + ev.repeatCount
                cortex_weights[ev.prevSym][ev.nextSym] += LR * strength

    # 1. Esperienza Online
    SEQ = [0, 1, 2, 3, 4]
    print(f"  Sequenza sperimentata (online): {' -> '.join([str(s) for s in SEQ])}")
    
    # Ripetiamo A->B molte volte per simulare un collo di bottiglia / evento frequente
    for _ in range(5):
        log_experience([0, 1])
        
    # Poi eseguiamo la sequenza completa
    log_experience(SEQ)
    
    print(f"  Eventi nel Ring Buffer dopo l'online: {buf.logSize} (Overflows: {buf.overflowCount})")
    for idx, ev in enumerate(buf.entries()):
        print(f"    - [{idx}] {ev.prevSym} -> {ev.nextSym} (Ripetizioni compresse: {ev.repeatCount})")
        
    # 2. Replay Offline
    CYCLES = 3
    print(f"  Avvio fase Offline: {CYCLES} cicli di replay notturno...")
    offline_replay(CYCLES)
    
    # 3. Verifica Pesi Appresi
    print("\n  Pesi consolidati nella corteccia (Matrice di transizione):")
    for i in range(N_SYM):
        row = " ".join([f"{w:5.2f}" for w in cortex_weights[i]])
        print(f"    Stato {i}: [{row}]")
        
    peso_ab = cortex_weights[0][1]
    peso_cd = cortex_weights[2][3]
    
    print(f"\n  Verifica Prioritized Replay:")
    print(f"    Peso A->B (vissuta 6 volte, compressa): {peso_ab:.2f}")
    print(f"    Peso C->D (vissuta 1 volta): {peso_cd:.2f}")
    
    if HAVE_PLT:
        import numpy as np
        fig, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(cortex_weights, cmap="Blues")
        for i in range(N_SYM):
            for j in range(N_SYM):
                if cortex_weights[i][j] > 0:
                    ax.text(j, i, f"{cortex_weights[i][j]:.1f}", ha="center", va="center")
        ax.set_title("Pesi Consolidati (Corteccia) via Replay")
        ax.set_xlabel("Stato Futuro")
        ax.set_ylabel("Stato Corrente")
        fig.colorbar(cax)
        fig.tight_layout()
        fig.savefig("./outputs/test_replay.png", dpi=130)
        print("  -> Grafico salvato in ./outputs/test_replay.png\n")
    else:
        print("  -> Matplotlib non disponibile, grafico saltato.\n")


def test_short_term_plasticity():
    print("--- Test 5: Short-Term Plasticity (STP - Coda Vescicolare) ---")
    syn = Synapse(weight=1.0)
    
    # 2 epoche: bassa frequenza, poi altissima frequenza (scarica continua)
    charges = []
    
    # Treno di 5 spike lenti (ogni 20 scan)
    print("  Stimolazione a bassa frequenza (recupero quasi totale):")
    for i in range(5):
        for _ in range(19): 
            syn.charge(False)
        ch = syn.charge(True)
        charges.append(ch)
        print(f"    Spike {i+1} -> charge: {ch:.1f} (Risorse: {syn.R:.2f})")
        
    # Treno di 5 spike veloci (ogni 1 scan, raffica)
    print("  Stimolazione ad alta frequenza (depressione/svuotamento):")
    for i in range(5):
        ch = syn.charge(True)
        charges.append(ch)
        print(f"    Spike {i+6} -> charge: {ch:.1f} (Risorse: {syn.R:.2f})")
        
    print("  -> Si osserva chiaramente la depressione sinaptica durante la raffica.\n")


def test_divisive_normalization():
    print("--- Test 6: Inibizione Divisiva (Shunting Inhibition) ---")
    n = Neuron("n", n_masters=2)
    
    # Diamo un drive eccitatorio massiccio che sfonderrebbe il CEILING rapidamente
    DRIVE = 800.0
    
    # Test A: Senza Shunting
    for _ in range(3):
        n.dendrite.deposit(0, DRIVE, TX_EXC | DRIVER)
        n.dendrite.run()
    acc_no_shunt = n.dendrite.accumulator
    
    n.dendrite.accumulator = 0.0 # reset
    
    # Test B: Con Shunting
    SHUNT_VAL = 2000.0
    for _ in range(3):
        n.dendrite.deposit(0, DRIVE, TX_EXC | DRIVER)
        n.dendrite.deposit(1, SHUNT_VAL, TX_SHUNT | DRIVER)
        n.dendrite.run()
    acc_shunt = n.dendrite.accumulator
    
    print(f"  Accumulo SENZA Shunting: {acc_no_shunt:.1f}")
    print(f"  Accumulo CON Shunting: {acc_shunt:.1f}")
    print("  -> L'inibizione shunting ha scalato l'accumulatore agendo da divisore.\n")


def test_nonlinear_dendrite():
    print("--- Test 7: Albero Dendritico Non Lineare (NMDA Spikes) ---")
    n = Neuron("n", n_masters=4)
    # n.dendrite ha LOCAL_TH = 1200.0, LOCAL_BOOST = 1500.0
    
    CH = 700.0 # Carica sub-soglia locale
    
    # Caso 1: Due segnali su rami DIVERSI (0 e 1)
    n.dendrite.deposit(0, CH, TX_EXC | DRIVER, branch_id=0)
    n.dendrite.deposit(1, CH, TX_EXC | DRIVER, branch_id=1)
    n.dendrite.run()
    acc_diversi = n.dendrite.accumulator
    
    n.dendrite.accumulator = 0.0 # reset
    
    # Caso 2: Due segnali sullo STESSO ramo (0)
    n.dendrite.deposit(0, CH, TX_EXC | DRIVER, branch_id=0)
    n.dendrite.deposit(1, CH, TX_EXC | DRIVER, branch_id=0)
    n.dendrite.run()
    acc_stesso = n.dendrite.accumulator
    
    print(f"  Input totale iniettato: {CH*2}")
    print(f"  Spike su rami separati (somma lineare): Accumulatore = {acc_diversi:.1f}")
    print(f"  Spike sullo stesso ramo (NMDA boost)  : Accumulatore = {acc_stesso:.1f}")
    print("  -> La coincidenza sullo stesso ramo scatena la pre-soglia locale (OR-of-ANDs).\n")


if __name__ == "__main__":
    import os
    os.makedirs("./outputs", exist_ok=True)
    test_debito_inibitorio()
    test_ei_oscillator()
    test_covariance_learning()
    test_replay_consolidation()
    test_short_term_plasticity()
    test_divisive_normalization()
    test_nonlinear_dendrite()
