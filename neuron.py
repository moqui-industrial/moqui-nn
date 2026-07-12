#!/usr/bin/env python3
# =============================================================================
#  neuron.py
#  UN unico Neuron in stile FB industriale che integra tutta la roadmap.
#  Progettato per essere istanziato ~1.000.000 di volte (memoria snella via
#  __slots__) e connesso agli altri nel NeuronNetFacade (il blackboard, come
#  DeviceFacade; orchestrazione come DeviceManager).
#
#  Lignaggio dai due soli primitivi PLC ammessi:
#    - Actuator (ActuatorCore): la FSM bistabile Disabled/EnablePhase/Enabled/
#      DisablePhase -> spike one-shot + refrattario.
#    - SignalMgmt / coda: l'accumulatore a fuga proporzionale (legge di Little,
#      L = lambda*W; fuga = membrana RC). Coincidenza per DECADIMENTO, non per clock.
#
#  Tutto il resto e' configurazione di questi due:
#    coincidenza (decadimento della coda) · soglia adattiva/HUNGRY (omeostasi) ·
#    tre fattori (coincidenza x modulatore x eligibilita') · neuromodulazione
#    (kGain/kPlasticity/kExplore letti dal blackboard) · dead time (1 scan assone).
#
#  NIENTE politica cablata (niente Main/RuleEngine): il comportamento emerge.
# =============================================================================

import random

# --- costanti anatomiche (recipe di bootstrap, non neuromodulazione) ---------
ENABLE_TIME = 2        # larghezza spike (scan)
DISABLE_TIME = 3       # refrattario (scan)
THR_MIN = 20.0         # soglia minima
TARGET_ACT = 0.05      # set-point omeostatico di attivita' (frazione di scan attivi)
HOMEO_RATE = 0.5       # velocita' dell'omeostasi (soglia insegue l'attivita')
ACT_RATE = 0.02        # costante della media di attivita'
ELIG_DECAY = 0.80      # decadimento della traccia di eligibilita'

# stati FSM (come ActuatorStatus)
S_DISABLED = 0
S_ENABLEPHASE = 1
S_ENABLED = 2
S_DISABLEPHASE = 3


class Neuron:
    """FB neurone: stato locale snello + una scansione. Istanziabile ~1M volte."""
    __slots__ = ("L", "mu", "status", "timer", "threshold",
                 "activity", "elig", "spike", "spikePrev")

    def __init__(self, tau_inertia=8.0, threshold=100.0):
        self.L = 0.0                       # coda/potenziale (Little: L = lambda*W)
        self.mu = 1.0 / tau_inertia        # fuga proporzionale (1/W, membrana RC)
        self.status = S_DISABLED           # FSM ActuatorCore
        self.timer = 0
        self.threshold = threshold         # soglia adattiva (omeostasi)
        self.activity = 0.0                # media di attivita' (per HUNGRY)
        self.elig = 0.0                    # traccia di eligibilita' (tre fattori)
        self.spike = False                 # uscita di questa scansione
        self.spikePrev = False             # uscita scansione precedente (dead time=1)

    def scan(self, drive, kGain, kExplore, noise):
        # ---- DENDRITE: coda di Little, fuga proporzionale (integra in continuo) ----
        self.L = self.L * (1.0 - self.mu) + drive
        if self.L < 0.0:
            self.L = 0.0
        self.spikePrev = self.spike        # l'assone consegna con 1 scan di ritardo
        self.spike = False

        # neuromodulazione: il gain abbassa la soglia efficace; l'esplora aggiunge rumore
        effThr = self.threshold / kGain if kGain > 0.0 else self.threshold
        driveEff = self.L + noise * kExplore

        # ---- SOMA: FSM ActuatorCore (spike one-shot + refrattario) ----
        st = self.status
        if st == S_DISABLED:
            if driveEff >= effThr:
                self.status = S_ENABLEPHASE
                self.timer = ENABLE_TIME
        elif st == S_ENABLEPHASE:
            if self.timer > 0:
                self.timer -= 1
            if self.timer == 0:
                self.status = S_ENABLED
        elif st == S_ENABLED:
            self.spike = True              # spike one-shot
            self.status = S_DISABLEPHASE
            self.timer = DISABLE_TIME
        elif st == S_DISABLEPHASE:
            if self.timer > 0:
                self.timer -= 1
            if self.timer == 0:
                self.status = S_DISABLED

        # ---- OMEOSTASI (HUNGRY): la soglia insegue il set-point di attivita' ----
        self.activity += ACT_RATE * ((1.0 if self.spike else 0.0) - self.activity)
        self.threshold += HOMEO_RATE * (self.activity - TARGET_ACT) * 100.0
        if self.threshold < THR_MIN:
            self.threshold = THR_MIN

        # ---- ELIGIBILITA' (per i tre fattori: ricorda l'attivita' recente) ----
        self.elig = self.elig * ELIG_DECAY + (1.0 if self.spike else 0.0)
        return self.spike


class NeuronNetFacade:
    """Blackboard + orchestratore (come DeviceFacade + DeviceManager).
    Tiene i neuroni, il cablaggio (sinapsi in array piatti) e il NeuromodConfig
    globale scritto dal task lento. Una scansione = InputSignalUpdate -> run -> learn."""
    __slots__ = ("neurons", "pre", "post", "w", "sign", "n",
                 "kGain", "kPlasticity", "kExplore", "modulator", "rng")

    def __init__(self, n_neurons, tau=8.0, threshold=100.0, seed=0):
        self.neurons = [Neuron(tau, threshold) for _ in range(n_neurons)]
        self.n = n_neurons
        # sinapsi in array piatti (CSR-like): memoria compatta, stile facade
        self.pre = []; self.post = []; self.w = []; self.sign = []
        # NeuromodConfig globale (scritto dal task lento, letto ogni scan)
        self.kGain = 1.0
        self.kPlasticity = 0.0
        self.kExplore = 0.0
        self.modulator = 0.0               # terzo fattore globale (RPE/dopamina)
        self.rng = random.Random(seed)

    def connect(self, pre, post, weight, excitatory=True):
        self.pre.append(pre); self.post.append(post)
        self.w.append(weight); self.sign.append(1.0 if excitatory else -1.0)

    def set_neuromod(self, kGain, kPlasticity, kExplore):
        # TASK LENTO: set atomico del regime (mai incrementale), come MqttParameterSub
        self.kGain = kGain; self.kPlasticity = kPlasticity; self.kExplore = kExplore

    def scan(self, ext_drive=None):
        neurons = self.neurons
        n = self.n
        # 1. DATA PLANE: drive di ogni neurone dagli spike in arrivo (ritardo 1 scan)
        drive = [0.0] * n if ext_drive is None else list(ext_drive)
        pre = self.pre; post = self.post; w = self.w; sign = self.sign
        for k in range(len(pre)):
            if neurons[pre[k]].spikePrev:          # l'assone consegna il pre di 1 scan fa
                drive[post[k]] += sign[k] * w[k]
        # 2. esegui ogni neurone (come DeviceManager chiama ogni Actuator una volta)
        kG = self.kGain; kE = self.kExplore
        rng = self.rng
        for i in range(n):
            noise = rng.gauss(0.0, 1.0) if kE > 0.0 else 0.0
            neurons[i].scan(drive[i], kG, kE, noise)
        # 3. CONTROL PLANE: apprendimento a tre fattori sulle sinapsi
        #    dw = kPlasticity * modulatore(globale) * elig_pre * spike_post
        kP = self.kPlasticity; mod = self.modulator
        if kP != 0.0 and mod != 0.0:
            for k in range(len(pre)):
                po = neurons[post[k]]
                if po.spike:
                    dw = kP * mod * neurons[pre[k]].elig
                    w[k] += dw
                    if w[k] < 0.0:
                        w[k] = 0.0

    def spikes(self):
        return [i for i in range(self.n) if self.neurons[i].spike]


# =============================================================================
#  DEMO 1 - correttezza: tutti i meccanismi insieme in una piccola rete
# =============================================================================
def demo_functional():
    print("DEMO 1 - correttezza: coincidenza + tre fattori + neuromodulazione\n")
    net = NeuronNetFacade(n_neurons=3, tau=8.0, threshold=100.0, seed=1)
    # neuroni 0,1 = ingressi; neurone 2 = uscita. 0->2 e 1->2 plastiche.
    net.connect(0, 2, weight=60.0, excitatory=True)
    net.connect(1, 2, weight=60.0, excitatory=True)

    def drive_inputs(a, b):
        # inietta corrente forte sugli ingressi per farli sparare
        return [300.0 if a else 0.0, 300.0 if b else 0.0, 0.0]

    # regime IMPARA: plasticita' + modulatore positivo (ricompensa)
    net.set_neuromod(kGain=1.0, kPlasticity=0.4, kExplore=0.0)
    net.modulator = 1.0
    w0_start = net.w[0]
    for t in range(60):
        coincident = (t % 4 == 0)         # i due ingressi sparano insieme, a ritmo
        net.scan(drive_inputs(coincident, coincident))
    w0_learn = net.w[0]
    print("  regime IMPARA (coincidenza + modulatore>0):")
    print("    peso 0->2: %.1f -> %.1f (delta %+.1f)  [rinforzo per coincidenza]"
          % (w0_start, w0_learn, w0_learn - w0_start))

    # regime ESEGUE: plasticita' spenta -> i pesi si congelano
    net.set_neuromod(kGain=1.0, kPlasticity=0.0, kExplore=0.0)
    w1_start = net.w[0]
    for t in range(60):
        net.scan(drive_inputs(t % 4 == 0, t % 4 == 0))
    w1_end = net.w[0]
    print("  regime ESEGUE (plasticita' spenta):")
    print("    peso 0->2: %.1f -> %.1f (delta %+.1f)  [congelato: neuromod off]"
          % (w1_start, w1_end, w1_end - w1_start))
    ok = (w0_learn > w0_start + 1.0) and (abs(w1_end - w1_start) < 1e-6)
    print("  TEST %s (impara per coincidenza sotto modulatore; congela senza plasticita')\n"
          % ("PASS" if ok else "FAIL"))


# =============================================================================
#  DEMO 2 - scala: istanziare ~1M neuroni e misurare memoria e tempo/scan
# =============================================================================
def demo_scale():
    import time, tracemalloc
    print("DEMO 2 - scala: istanziazione di 1.000.000 di Neuron\n")
    N = 1_000_000
    tracemalloc.start()
    t0 = time.time()
    net = NeuronNetFacade(n_neurons=N, tau=8.0, threshold=100.0, seed=0)
    t_build = time.time() - t0
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print("  istanziati %d neuroni in %.2f s" % (N, t_build))
    print("  memoria: %.0f MB (%.0f byte/neurone con __slots__)"
          % (peak / 1e6, peak / N))
    # alcune scansioni dell'intera popolazione (drive esterno casuale sostenuto)
    import statistics as st
    scan_times = []
    active = 0
    for s in range(6):
        ext = [net.rng.uniform(0, 260) for _ in range(N)]
        t0 = time.time()
        net.scan(ext)
        scan_times.append(time.time() - t0)
        active = sum(1 for nn in net.neurons if nn.spike)
    t_scan = st.mean(scan_times)
    print("  scansione di 1M neuroni: %.2f s media  (%.2f us/neurone)"
          % (t_scan, t_scan / N * 1e6))
    print("  neuroni attivi (ultima scansione, dopo che l'FSM ha completato l'enable): %d" % active)
    print("\n  -> 1M istanze stanno in memoria e girano. Per il real-time a 1M si")
    print("     vettorizza (numpy) o si scende sul PLC/C: questo FB e' il modello")
    print("     di riferimento per-istanza, la logica e' la stessa vettorizzata.")


def main():
    print("=" * 70)
    print("Neuron unico (stile FB industriale) - tutta la roadmap in un file")
    print("=" * 70 + "\n")
    demo_functional()
    demo_scale()


if __name__ == "__main__":
    main()
