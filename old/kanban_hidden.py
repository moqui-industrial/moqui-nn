#!/usr/bin/env python3
# =============================================================================
#  kanban_hidden.py
#  Lo strato nascosto come sistema KANBAN PULL.
#   - ogni nascosto (slave) TIRA dai master a monte: HUNGRY abbassa la soglia.
#   - la Synapse e' il BIN kanban per-master: peso = priorita' della carta.
#   - PROBED = la carta in attesa (master arrivato in finestra).
#   - competizione k-WTA tra nascosti -> ognuno si specializza su un pattern
#     (apprendimento competitivo: assegna detector distinti a input distinti).
#   - apprendimento: al proprio spike, rinforza i bin con carta (PROBED);
#     normalizzazione omeostatica (vincolo aggregato sul WIP totale).
#  Ipotesi da verificare: l'auto-organizzazione da' 4 codici DISTINTI sui 4
#  pattern one-hot (la base separabile per XOR), SENZA supervisione.
# =============================================================================

import sys, random
sys.path.insert(0, "/home/claude")
from signalmgmt_mc import SignalMgmtMC, TX_EXC, DRIVER, E_SITE


# --- soma minimale: spike one-shot + refrattario ---
class Soma:
    def __init__(self):
        self.SPIKE_WIDTH = 2
        self.REFRACTORY = 3
        self.enableRequest = False
        self.enable = False
        self.enabledSensor = False
        self.status = 0   # 0 Disabled,1 EnablePhase,2 Enabled,3 DisablePhase
        self.st = 0
        self.rt = 0

    def run(self):
        self.enable = False
        self.enabledSensor = False
        if self.status == 0:
            if self.enableRequest:
                self.status = 1; self.st = self.SPIKE_WIDTH
        elif self.status == 1:
            if self.st > 0: self.st -= 1
            if self.st == 0: self.status = 2
        elif self.status == 2:
            self.enable = True; self.enabledSensor = True
            self.status = 3; self.rt = self.REFRACTORY
        elif self.status == 3:
            if self.rt > 0: self.rt -= 1
            if self.rt == 0: self.status = 0


class Hidden:
    def __init__(self, n_masters):
        self.dend = SignalMgmtMC(n_masters)
        self.soma = Soma()
        # un bin kanban per master: peso (priorita' della carta)
        self.w = [1.0 for _ in range(n_masters)]


N_IN = 4
N_HID = 6
QUANTUM = 200.0
LR = 0.30
SUM_W = 4.0           # WIP totale per nascosto (vincolo aggregato omeostatico)
PRESENT = 10
KWTA = 1              # quanti nascosti possono vincere per trial


def normalize(h):
    s = sum(h.w)
    if s > 1e-6:
        f = SUM_W / s
        for m in range(len(h.w)):
            h.w[m] *= f


def build(rng):
    hid = [Hidden(N_IN) for _ in range(N_HID)]
    for h in hid:
        for m in range(N_IN):
            h.w[m] = 1.0 + rng.uniform(-0.5, 0.5)   # jitter: rompe la simmetria
        normalize(h)
        h.dend.THRESHOLD = 600.0      # regime di coincidenza (no saturazione)
        h.dend.LEAK_RATE = 60.0
    return hid


PATTERNS = [(0, 0), (0, 1), (1, 0), (1, 1)]


def present(hid, pat, rng, learn):
    # input one-hot: pattern (b0,b1) accende I0_b0 e I1_b1 -> bit master 0..3
    b0, b1 = pat
    active = [0 + b0, 2 + b1]          # master bits attivi
    # reset dinamico
    for h in hid:
        h.dend.accumulator = 0.0
        h.dend.probe_ring = [0] * len(h.dend.probe_ring)
        h.dend.probed = 0
        h.soma.status = 0; h.soma.st = 0; h.soma.rt = 0
        h.soma.enable = False; h.soma.enabledSensor = False

    fired_count = {i: 0 for i in range(N_HID)}
    c = 0
    while c < PRESENT:
        # 1. ogni input attivo deposita nel bin del proprio master in ogni nascosto
        for i, h in enumerate(hid):
            for m in active:
                # MC_CamIn: ingaggia il master m con priorita' = peso del bin
                h.dend.deposit(master_bit=m, charge=h.w[m] * QUANTUM, tag=TX_EXC | DRIVER)
        # 2. esegui i dendriti (aggiorna accumulatore, PROBED, HUNGRY)
        for h in hid:
            h.dend.run()
        # 3. k-WTA: solo i top-KWTA per accumulatore possono sparare (competizione)
        order = sorted(range(N_HID), key=lambda i: hid[i].dend.accumulator, reverse=True)
        winners = set(order[:KWTA])
        for i, h in enumerate(hid):
            h.soma.enableRequest = h.dend.enableRequest and (i in winners)
            h.soma.run()
            h.dend.enabledSensor = h.soma.enabledSensor
            if h.soma.enable:
                fired_count[i] += 1
                # 4. APPRENDIMENTO: al mio spike, rinforza i bin con CARTA (PROBED)
                if learn:
                    for m in range(N_IN):
                        if (h.dend.probed >> m) & 1:
                            h.w[m] += LR
                    normalize(h)
        c += 1
    return fired_count


def code(hid, pat, rng):
    fc = present(hid, pat, rng, learn=False)
    return tuple(1 if fc[i] > 0 else 0 for i in range(N_HID))


def main():
    import numpy as np
    print("Auto-organizzazione competitiva dello strato nascosto (kanban pull)\n")
    for seed in range(6):
        rng = random.Random(seed)
        hid = build(rng)
        # training NON supervisionato: presenta i pattern, i nascosti competono e tirano
        for t in range(400):
            pat = PATTERNS[rng.randrange(4)]
            present(hid, pat, rng, learn=True)
        # codici finali (quale nascosto vince per ogni pattern)
        codes = {}
        for pat in PATTERNS:
            codes[pat] = code(hid, pat, rng)
        distinct = len(set(codes.values()))
        # XOR separabile su questi codici?
        V = np.array([list(codes[p]) for p in PATTERNS], float)
        lab = np.array([1.0 if (p[0] ^ p[1]) else -1.0 for p in PATTERNS])
        w = np.zeros(N_HID); bv = 0.0
        for _ in range(3000):
            for i in range(4):
                if np.sign(w @ V[i] + bv) != lab[i]:
                    w += lab[i] * V[i]; bv += lab[i]
        sep = np.mean(np.sign(V @ w + bv) == lab)
        winners = {p: [i for i in range(N_HID) if codes[p][i]] for p in PATTERNS}
        print("seed %d: codici distinti %d/4, XOR-sep %.2f | vincitori per pattern: %s"
              % (seed, distinct, sep, {("%d%d" % p): winners[p] for p in PATTERNS}))


if __name__ == "__main__":
    main()
