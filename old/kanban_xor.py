#!/usr/bin/env python3
# =============================================================================
#  kanban_xor.py
#  XOR end-to-end sull'architettura KANBAN PULL.
#   Fase A: lo strato nascosto si auto-organizza NON supervisionato (pull +
#           competizione + inseguimento). Da' codici distinti/separabili.
#   Fase B: il readout impara per CORREZIONE D'ERRORE (perceptron/delta = la
#           fibra rampicante cerebellare). Niente backprop: errore locale a
#           uno strato sopra una rappresentazione gia' separabile.
#  Tutto lo strato nascosto e' fatto di Neuron veri (SignalMgmtMC + Soma).
# =============================================================================

import sys, random
sys.path.insert(0, "/home/claude")
import numpy as np
from kanban_hidden import (build, present, PATTERNS, N_HID, N_IN, normalize)


def hidden_code(hid, pat, rng):
    fc = present(hid, pat, rng, learn=False)
    return [1.0 if fc[i] > 0 else 0.0 for i in range(N_HID)]


def train_one(seed):
    rng = random.Random(seed)

    # --- FASE A: auto-organizzazione NON supervisionata, con RIPROVA su collisione ---
    #   se i 4 pattern non ottengono codici distinti, la configurazione non
    #   differenzia: la si rifa' (potatura/riformazione dello sviluppo).
    for attempt in range(8):
        hid = build(rng)
        for t in range(500):
            present(hid, PATTERNS[rng.randrange(4)], rng, learn=True)
        codes = {p: tuple(hidden_code(hid, p, rng)) for p in PATTERNS}
        if len(set(codes.values())) == 4:      # 4 codici distinti -> ok
            break
    codes = {p: list(codes[p]) for p in PATTERNS}

    # --- FASE B: readout a correzione d'errore (perceptron) sui codici nascosti ---
    W = [[0.5 for _ in range(N_HID)] for _ in range(2)]   # 2 uscite x N_HID
    LR = 0.10
    for t in range(2000):
        p = PATTERNS[rng.randrange(4)]
        x = codes[p]
        tgt = (p[0] ^ p[1])                      # classe XOR
        drive = [sum(W[o][h] * x[h] for h in range(N_HID)) for o in range(2)]
        pred = 0 if drive[0] >= drive[1] else 1
        if pred != tgt:                          # aggiorna solo su errore (fibra rampicante)
            for h in range(N_HID):
                W[tgt][h] += LR * x[h]
                W[1 - tgt][h] -= LR * x[h]
                if W[tgt][h] > 6.0: W[tgt][h] = 6.0
                if W[1 - tgt][h] < 0.0: W[1 - tgt][h] = 0.0

    # --- valutazione XOR ---
    ok = 0
    for p in PATTERNS:
        x = codes[p]
        drive = [sum(W[o][h] * x[h] for h in range(N_HID)) for o in range(2)]
        pred = 0 if drive[0] >= drive[1] else 1
        if pred == (p[0] ^ p[1]):
            ok += 1
    return ok / 4.0, codes


def main():
    print("XOR end-to-end - architettura kanban pull (nascosti veri + readout a errore)\n")
    res = []
    for seed in range(12):
        acc, codes = train_one(seed)
        res.append(acc)
        mark = "  <-- XOR risolto" if acc >= 0.999 else ""
        print("seed %2d -> XOR %d/4  (%.2f)%s" % (seed, int(round(acc * 4)), acc, mark))
    solved = sum(1 for a in res if a >= 0.999)
    print("\nrisolti 4/4: %d/%d seed   accuratezza media %.3f" % (solved, len(res), np.mean(res)))
    print("\nStrato nascosto: Neuron veri (SignalMgmtMC), auto-organizzati col pull")
    print("kanban (HUNGRY) + competizione + inseguimento (PROBED). Nessuna backprop.")


if __name__ == "__main__":
    main()
