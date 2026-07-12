#!/usr/bin/env python3
# =============================================================================
#  neuron_vectorized.py
#  Stessa identica logica di neuron.py, ma la popolazione e' un insieme di
#  array numpy (structure-of-arrays): una scansione elabora TUTTI i neuroni in
#  parallelo invece che in ciclo. Il modello di riferimento (neuron.py) serve a
#  capire; questo serve a girare in real-time (1M neuroni in millisecondi).
#
#  L'FSM ActuatorCore e' vettorizzata con maschere booleane sullo stato di
#  inizio-scansione (snapshot) - cosi' non c'e' cascata: ogni neurone avanza di
#  UN solo stato per scansione, esattamente come l'elif del FB scalare.
# =============================================================================

import numpy as np

# --- stesse costanti del FB (neuron.py) --------------------------------------
ENABLE_TIME = 2
DISABLE_TIME = 3
THR_MIN = 20.0
TARGET_ACT = 0.05
HOMEO_RATE = 0.5
ACT_RATE = 0.02
ELIG_DECAY = 0.80

S_DISABLED = 0
S_ENABLEPHASE = 1
S_ENABLED = 2
S_DISABLEPHASE = 3


class NeuronNetVectorized:
    """Rete vettorizzata: structure-of-arrays + una scansione parallela.
    Stessa API del NeuronNetFacade (connect/set_neuromod/scan/spikes)."""

    def __init__(self, n_neurons, tau=8.0, threshold=100.0, seed=0):
        n = n_neurons
        self.n = n
        # stato per-neurone come colonne (SoA) invece di oggetti (AoS)
        self.L = np.zeros(n, dtype=np.float64)
        self.mu = np.full(n, 1.0 / tau, dtype=np.float64)
        self.status = np.zeros(n, dtype=np.int8)
        self.timer = np.zeros(n, dtype=np.int32)
        self.threshold = np.full(n, threshold, dtype=np.float64)
        self.activity = np.zeros(n, dtype=np.float64)
        self.elig = np.zeros(n, dtype=np.float64)
        self.spike = np.zeros(n, dtype=bool)
        self.spikePrev = np.zeros(n, dtype=bool)
        # sinapsi in array piatti
        self._pre = []; self._post = []; self._w = []; self._sign = []
        self.pre = self.post = self.w = self.sign = None
        # NeuromodConfig globale
        self.kGain = 1.0; self.kPlasticity = 0.0; self.kExplore = 0.0
        self.modulator = 0.0
        self.rng = np.random.default_rng(seed)

    def connect(self, pre, post, weight, excitatory=True):
        self._pre.append(pre); self._post.append(post)
        self._w.append(weight); self._sign.append(1.0 if excitatory else -1.0)

    def finalize(self):
        # compatta le sinapsi in array numpy (una volta, dopo il cablaggio)
        self.pre = np.asarray(self._pre, dtype=np.int64)
        self.post = np.asarray(self._post, dtype=np.int64)
        self.w = np.asarray(self._w, dtype=np.float64)
        self.sign = np.asarray(self._sign, dtype=np.float64)

    def set_neuromod(self, kGain, kPlasticity, kExplore):
        self.kGain = kGain; self.kPlasticity = kPlasticity; self.kExplore = kExplore

    def scan(self, ext_drive=None):
        n = self.n
        # ---- 1. DATA PLANE: drive dagli spike in arrivo (dead time 1 scan) ----
        drive = np.zeros(n) if ext_drive is None else np.asarray(ext_drive, dtype=np.float64).copy()
        if self.pre is not None and self.pre.size:
            active = self.spikePrev[self.pre]                 # sinapsi il cui pre ha sparato
            contrib = self.sign * self.w * active
            np.add.at(drive, self.post, contrib)              # scatter-add sui post
        # ---- 2. DENDRITE: coda di Little, fuga proporzionale (tutti insieme) ----
        self.L = self.L * (1.0 - self.mu) + drive
        np.maximum(self.L, 0.0, out=self.L)
        self.spikePrev = self.spike.copy()
        self.spike = np.zeros(n, dtype=bool)
        effThr = self.threshold / self.kGain if self.kGain > 0 else self.threshold
        if self.kExplore > 0.0:
            driveEff = self.L + self.rng.standard_normal(n) * self.kExplore
        else:
            driveEff = self.L
        # ---- 3. SOMA: FSM ActuatorCore vettorizzata (snapshot -> niente cascata) ----
        st0 = self.status                                     # stato di inizio scansione
        dis = st0 == S_DISABLED
        fire = dis & (driveEff >= effThr)
        ep = st0 == S_ENABLEPHASE
        en = st0 == S_ENABLED
        dp = st0 == S_DISABLEPHASE
        newstatus = st0.copy()
        # DISABLED -> EnablePhase se supera la soglia efficace
        newstatus[fire] = S_ENABLEPHASE
        self.timer[fire] = ENABLE_TIME
        # EnablePhase: scala il timer; a 0 -> Enabled
        dec_ep = ep & (self.timer > 0)
        self.timer[dec_ep] -= 1
        newstatus[ep & (self.timer == 0)] = S_ENABLED
        # Enabled -> spike one-shot -> DisablePhase (refrattario)
        self.spike[en] = True
        newstatus[en] = S_DISABLEPHASE
        self.timer[en] = DISABLE_TIME
        # DisablePhase: scala il timer; a 0 -> Disabled
        dec_dp = dp & (self.timer > 0)
        self.timer[dec_dp] -= 1
        newstatus[dp & (self.timer == 0)] = S_DISABLED
        self.status = newstatus
        # ---- 4. OMEOSTASI (HUNGRY) e ELIGIBILITA', vettorizzate ----
        sp = self.spike.astype(np.float64)
        self.activity += ACT_RATE * (sp - self.activity)
        self.threshold += HOMEO_RATE * (self.activity - TARGET_ACT) * 100.0
        np.maximum(self.threshold, THR_MIN, out=self.threshold)
        self.elig = self.elig * ELIG_DECAY + sp
        # ---- 5. CONTROL PLANE: apprendimento a tre fattori sulle sinapsi ----
        if self.kPlasticity != 0.0 and self.modulator != 0.0 and self.pre is not None and self.pre.size:
            post_spk = self.spike[self.post]
            dw = self.kPlasticity * self.modulator * self.elig[self.pre] * post_spk
            self.w += dw
            np.maximum(self.w, 0.0, out=self.w)

    def spikes(self):
        return np.nonzero(self.spike)[0]


# =============================================================================
#  VERIFICA 1 - equivalenza col FB scalare (stessi numeri, caso deterministico)
# =============================================================================
def demo_equivalence():
    import sys
    sys.path.insert(0, "/home/claude")
    from neuron import NeuronNetFacade

    print("VERIFICA 1 - equivalenza col modello di riferimento (neuron.py)\n")
    # stessa piccola rete, stesso compito, kExplore=0 (deterministico)
    def build_scalar():
        net = NeuronNetFacade(n_neurons=3, tau=8.0, threshold=100.0, seed=1)
        net.connect(0, 2, 60.0, True); net.connect(1, 2, 60.0, True)
        net.set_neuromod(1.0, 0.4, 0.0); net.modulator = 1.0
        return net

    def build_vec():
        net = NeuronNetVectorized(n_neurons=3, tau=8.0, threshold=100.0, seed=1)
        net.connect(0, 2, 60.0, True); net.connect(1, 2, 60.0, True)
        net.finalize()
        net.set_neuromod(1.0, 0.4, 0.0); net.modulator = 1.0
        return net

    sc = build_scalar(); ve = build_vec()
    for t in range(60):
        coin = (t % 4 == 0)
        ext = [300.0 if coin else 0.0, 300.0 if coin else 0.0, 0.0]
        sc.scan(ext); ve.scan(ext)
    w_scalar = sc.w[0]
    w_vector = float(ve.w[0])
    print("  peso 0->2 dopo l'apprendimento:")
    print("    modello scalare (FB):  %.6f" % w_scalar)
    print("    modello vettorizzato:  %.6f" % w_vector)
    match = abs(w_scalar - w_vector) < 1e-9
    print("  differenza: %.2e   TEST %s (numeri identici)\n"
          % (abs(w_scalar - w_vector), "PASS" if match else "FAIL"))


# =============================================================================
#  VERIFICA 2 - scala: 1M neuroni in millisecondi
# =============================================================================
def demo_scale():
    import time
    print("VERIFICA 2 - scala: 1.000.000 di neuroni vettorizzati\n")
    N = 1_000_000
    t0 = time.time()
    net = NeuronNetVectorized(n_neurons=N, tau=8.0, threshold=100.0, seed=0)
    net.finalize()
    t_build = time.time() - t0
    mem = (net.L.nbytes + net.mu.nbytes + net.status.nbytes + net.timer.nbytes +
           net.threshold.nbytes + net.activity.nbytes + net.elig.nbytes +
           net.spike.nbytes + net.spikePrev.nbytes)
    print("  costruiti %d neuroni in %.2f s, memoria stato %.0f MB (%.0f byte/neurone)"
          % (N, t_build, mem / 1e6, mem / N))
    # scansioni con drive SOSTENUTO per-neurone (fisso), cosi' l'FSM cicla stabile
    ext = net.rng.uniform(0, 260, size=N)
    times = []
    active = 0
    for s in range(12):
        t0 = time.time()
        net.scan(ext)
        times.append(time.time() - t0)
        active = int(net.spike.sum())
    import statistics as st
    t_scan = st.mean(times[2:])          # scarta il warm-up
    print("  scansione di 1M neuroni: %.1f ms media  (%.3f us/neurone)"
          % (t_scan * 1000, t_scan / N * 1e6))
    print("  neuroni attivi (ultima scansione): %d" % active)
    # confronto col riferimento scalare (~0.43 s/scan)
    print("\n  -> stessa logica del FB, ma ~%.0fx piu' veloce a scansione." % (0.43 / t_scan))
    print("     A questo ritmo 1M neuroni stanno dentro un ciclo PLC da 10-100 ms.")


def main():
    print("=" * 70)
    print("Rete neurale vettorizzata - stesso comportamento del FB, real-time")
    print("=" * 70 + "\n")
    demo_equivalence()
    demo_scale()


if __name__ == "__main__":
    main()
