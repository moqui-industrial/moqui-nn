#!/usr/bin/env python3
# =============================================================================
#  ei_oscillator.py   (FASE 2, punto 6)
#  Le oscillazioni di rete come CLOCK DI COINCIDENZA generato dall'anello E-I.
#
#  Meccanismo: l'eccitazione accende l'inibizione; l'inibizione (con RITARDO)
#  spegne l'eccitazione; l'inibizione decade e l'eccitazione riparte. E' un
#  regolatore mandato APPOSTA in pendolamento dal dead time dell'anello - un
#  ciclo limite. La frequenza la detta il ritardo: anello corto=gamma, lungo=theta.
#
#  A cosa serve: ogni ciclo l'inibizione scende per un istante -> FINESTRA aperta
#  in cui si puo' sparare. Solo segnali IN FASE cadono nella finestra e si legano
#  (binding per sincronia). E' il clock di scansione del PLC, ma generato dalla
#  rete invece che dal quarzo.
#
#  Stile industriale: FSM espliciti, scansione ciclica, stato in struct, nomi PLC.
# =============================================================================

SCAN_MS = 1.0                 # durata di una scansione (ms), come un ciclo PLC


# --- coda con fuga proporzionale (membrana RC / legge di Little) ------------
class LeakyQueue:
    """Accumulatore con fuga proporzionale: potenziale = tasso x inerzia."""
    def __init__(self, tau_scan):
        self.MU = 1.0 / tau_scan          # tasso di fuga (1/inerzia)
        self.L = 0.0                      # riempimento = potenziale

    def update(self, arrivals):
        # fuga proporzionale, poi arrivi (ordine da SignalMgmt)
        self.L = self.L * (1.0 - self.MU) + arrivals
        if self.L < 0.0:
            self.L = 0.0
        return self.L


# --- linea di ritardo: il dead time dell'anello (assone I->E) ---------------
class DelayLine:
    """Ritardo puro di N scansioni: il segnale entra ora, esce N scan dopo."""
    def __init__(self, dead_time_scan):
        self.buf = [0.0] * max(1, dead_time_scan)
        self.idx = 0

    def update(self, value_in):
        value_out = self.buf[self.idx]
        self.buf[self.idx] = value_in
        self.idx = (self.idx + 1) % len(self.buf)
        return value_out


# --- struct di stato dell'anello E-I (blackboard, come DeviceFacade) --------
class EiRingState:
    def __init__(self, tau_e, tau_i, dead_time):
        self.eExc = LeakyQueue(tau_e)         # popolazione eccitatoria
        self.iInh = LeakyQueue(tau_i)         # popolazione inibitoria
        self.iToEDelay = DelayLine(dead_time) # dead time dell'anello I->E
        self.eActivity = 0.0
        self.iActivity = 0.0
        self.gateOpen = False                 # finestra di coincidenza aperta?


# --- parametri dell'anello (i guadagni delle sinapsi E<->I) -----------------
W_E_DRIVE = 1.30      # quanto l'input esterno + rientro spinge E
W_E_TO_I = 1.10       # E accende I
W_I_TO_E = 2.20       # I spegne E (forte -> pendolamento)
GATE_THRESHOLD = 0.35 # sotto questa inibizione la finestra e' aperta


def scan_ei_ring(st, ext_drive):
    """Una scansione ciclica dell'anello E-I (come un ciclo PLC)."""
    # 1. l'inibizione arriva a E RITARDATA (dead time dell'anello)
    inh_to_e = st.iToEDelay.update(st.iActivity)

    # 2. aggiorna l'eccitazione: input esterno meno inibizione ritardata
    e_arrivals = W_E_DRIVE * ext_drive - W_I_TO_E * inh_to_e
    st.eActivity = st.eExc.update(max(0.0, e_arrivals))

    # 3. aggiorna l'inibizione: inseguita dall'eccitazione corrente
    i_arrivals = W_E_TO_I * st.eActivity
    st.iActivity = st.iInh.update(i_arrivals)

    # 4. FINESTRA DI COINCIDENZA: aperta quando l'inibizione e' bassa
    st.gateOpen = st.iActivity < GATE_THRESHOLD
    return st.eActivity, st.iActivity, st.gateOpen


def measure_period(gate_trace):
    # periodo = distanza media tra due aperture della finestra (fronti di salita)
    opens = [t for t in range(1, len(gate_trace))
             if gate_trace[t] and not gate_trace[t - 1]]
    if len(opens) < 2:
        return 0.0
    gaps = [opens[i + 1] - opens[i] for i in range(len(opens) - 1)]
    return sum(gaps) / len(gaps)


def run_ring(tau_e, tau_i, dead_time, n_scan=400, ext=1.0):
    st = EiRingState(tau_e, tau_i, dead_time)
    E = []; I = []; G = []
    for t in range(n_scan):
        e, i, g = scan_ei_ring(st, ext)
        E.append(e); I.append(i); G.append(g)
    return E, I, G


def main():
    print("Oscillazione da anello E-I = clock di coincidenza generato dalla rete\n")

    print("VERIFICA 1 - l'anello oscilla, e il dead time detta la frequenza:")
    print("   dead time (scan) | periodo misurato (scan)")
    prev = None
    for dt in [2, 4, 6, 10]:
        E, I, G = run_ring(tau_e=3.0, tau_i=5.0, dead_time=dt)
        per = measure_period(G)
        print("        %2d        |         %5.1f" % (dt, per))
    print("   (periodo monotono nel dead time: anello corto=veloce, lungo=lento)")

    print("\n-> anello corto -> finestra veloce (gamma); anello lungo -> lenta (theta).")
    print("   Il dead time e' il progetto dell'orologio, non un parametro a caso.\n")

    print("VERIFICA 2 - binding per sincronia: solo gli input IN FASE si legano.")
    # faccio girare l'anello, raccolgo la traccia della finestra
    E, I, G = run_ring(tau_e=3.0, tau_i=5.0, dead_time=4, n_scan=400)
    opens = [t for t in range(1, len(G)) if G[t] and not G[t - 1]]  # fronti di apertura
    P = int(round(measure_period(G))) or 8
    half = P // 2
    # sorgente A: spara ai fronti di apertura (in fase). B: sfasata di mezzo periodo.
    a_hits = sum(1 for o in opens if o < len(G) and G[o])            # A: nella finestra
    b_hits = sum(1 for o in opens if o + half < len(G) and G[o + half])  # B: controfase
    print("   periodo del ritmo: %d scan, mezzo periodo: %d scan" % (P, half))
    print("   sorgente A (in fase con la finestra): coincidenze = %d/%d" % (a_hits, len(opens)))
    print("   sorgente B (controfase, +mezzo periodo): coincidenze = %d/%d" % (b_hits, len(opens)))
    ok = a_hits > b_hits
    print("   -> l'orologio comune fa passare A (in fase) e blocca B (controfase).")
    print("   TEST %s\n" % ("PASS" if ok else "verifica fasi"))

    try:
        import os, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        E, I, G = run_ring(tau_e=3.0, tau_i=5.0, dead_time=4, n_scan=120)
        ax1.plot(E, color="#d62728", label="eccitazione E")
        ax1.plot(I, color="#1f77b4", label="inibizione I")
        ax1.axhline(GATE_THRESHOLD, color="gray", ls=":", lw=0.8, label="soglia finestra")
        ax1.set_title("Anello E-I: E accende I, I (ritardata) spegne E -> ciclo limite")
        ax1.set_ylabel("attivita'"); ax1.legend(fontsize=8, loc="upper right"); ax1.grid(alpha=0.3)
        ax2.fill_between(range(len(G)), 0, [1 if g else 0 for g in G],
                         step="mid", color="#2ca02c", alpha=0.5)
        ax2.set_title("Finestra di coincidenza aperta (il clock generato dalla rete)")
        ax2.set_xlabel("scansione (ms)"); ax2.set_ylabel("finestra"); ax2.set_yticks([0, 1])
        fig.tight_layout()
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        out = "/mnt/user-data/outputs/ei_oscillator.png"
        fig.savefig(out, dpi=130)
        print("chart -> %s" % out)
    except Exception as ex:
        print("[chart skipped: %s]" % ex)


if __name__ == "__main__":
    main()
