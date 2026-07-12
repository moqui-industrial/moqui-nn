#!/usr/bin/env python3
# =============================================================================
#  sequence_learning.py   (FASE 3, punto 7)
#  Apprendimento di sequenze sul RING BUFFER della storia recente - lo stesso
#  ring buffer del logger di stati del PLC (log4j di numeri e stati, non testo).
#
#  Idea (Igor): il log degli stati E' gia' una sequenza. Il ring buffer conserva
#  l'ORDINE per costruzione: A alla posizione t, B a t+1, C a t+2. Imparare la
#  sequenza = imparare che quella posizione relativa nel buffer si ripete.
#
#  Meccanismo: ring buffer degli eventi recenti (chi ha sparato nelle ultime N
#  scansioni) + regola che rinforza il legame consecutivo (STDP con offset: se
#  A era nel buffer e ora arriva B -> rinforza A->B). Predizione = visto A,
#  pre-attiva B prima che arrivi. Coincidenza-con-offset, gia' nei primitivi.
#
#  Stile industriale: ring buffer, scansione ciclica, matrice dei legami come
#  bitfield di pesi, FSM di lettura.
# =============================================================================

N_SYM = 6                 # simboli/stati distinti (A..F)
HIST_DEPTH = 3            # profondita' del ring buffer (quanto indietro guarda)
LR = 0.25                 # tasso di rinforzo del legame
DECAY_OLDER = 0.6         # un evento piu' vecchio nel buffer pesa meno (recency)


class HistoryRingBuffer:
    """Il ring buffer del logger di stati: tiene gli ultimi HIST_DEPTH eventi."""
    def __init__(self, depth):
        self.buf = [-1] * depth      # -1 = vuoto
        self.idx = 0

    def push(self, symbol):
        self.buf[self.idx] = symbol
        self.idx = (self.idx + 1) % len(self.buf)

    def recent(self):
        # restituisce (simbolo, eta') dal piu' recente al piu' vecchio
        out = []
        for age in range(1, len(self.buf) + 1):
            j = (self.idx - age) % len(self.buf)
            if self.buf[j] >= 0:
                out.append((self.buf[j], age))
        return out


class SequenceLearner:
    def __init__(self):
        # matrice dei legami: link[a][b] = quanto A predice B (peso)
        self.link = [[0.0] * N_SYM for _ in range(N_SYM)]
        self.hist = HistoryRingBuffer(HIST_DEPTH)

    def observe(self, symbol):
        # rinforza il legame da ogni evento recente nel buffer VERSO il corrente
        for (prev, age) in self.hist.recent():
            self.link[prev][symbol] += LR * (DECAY_OLDER ** (age - 1))
        self.hist.push(symbol)

    def predict(self, symbol):
        # dato un simbolo, qual e' il prossimo piu' legato?
        row = self.link[symbol]
        best = max(range(N_SYM), key=lambda b: row[b])
        return best if row[best] > 0 else -1

    def complete(self, seed, length):
        # completa la sequenza partendo da un innesco (predizione a catena)
        out = [seed]
        cur = seed
        for _ in range(length - 1):
            nxt = self.predict(cur)
            if nxt < 0:
                break
            out.append(nxt)
            cur = nxt
        return out


def sym(i):
    return "ABCDEF"[i]


def seq_str(seq):
    return "->".join(sym(s) for s in seq)


def main():
    print("Apprendimento di sequenze sul ring buffer del logger di stati\n")

    # sequenza da imparare: A->B->C->D->E (ordine fisso, ripetuto)
    SEQ = [0, 1, 2, 3, 4]
    print("Sequenza da imparare: %s (ripetuta durante l'addestramento)\n" % seq_str(SEQ))

    learner = SequenceLearner()
    # addestramento: scorri la sequenza molte volte
    for epoch in range(40):
        learner.hist = HistoryRingBuffer(HIST_DEPTH)   # reset del buffer a inizio sequenza
        for s in SEQ:
            learner.observe(s)

    print("VERIFICA 1 - predizione del prossimo elemento:")
    ok_pred = 0
    for i in range(len(SEQ) - 1):
        p = learner.predict(SEQ[i])
        good = (p == SEQ[i + 1])
        ok_pred += good
        print("   visto %s -> predice %s   (vero %s) %s"
              % (sym(SEQ[i]), sym(p) if p >= 0 else "-", sym(SEQ[i + 1]),
                 "OK" if good else "NO"))
    print("   TEST %s\n" % ("PASS" if ok_pred == len(SEQ) - 1 else "FAIL"))

    print("VERIFICA 2 - completamento da innesco parziale:")
    completed = learner.complete(SEQ[0], len(SEQ))
    print("   innesco '%s' -> completa: %s" % (sym(SEQ[0]), seq_str(completed)))
    good_complete = (completed == SEQ)
    print("   (sequenza vera: %s) TEST %s\n"
          % (seq_str(SEQ), "PASS" if good_complete else "FAIL"))

    print("VERIFICA 3 - innesco da meta' sequenza:")
    mid = learner.complete(SEQ[2], len(SEQ) - 2)
    print("   innesco '%s' -> completa: %s (atteso %s)"
          % (sym(SEQ[2]), seq_str(mid), seq_str(SEQ[2:])))
    print("   TEST %s\n" % ("PASS" if mid == SEQ[2:] else "FAIL"))

    # matrice dei legami appresa (per il grafico)
    try:
        import os, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        M = np.array(learner.link)
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(M, cmap="Blues", aspect="equal")
        ax.set_xticks(range(N_SYM)); ax.set_yticks(range(N_SYM))
        ax.set_xticklabels([sym(i) for i in range(N_SYM)])
        ax.set_yticklabels([sym(i) for i in range(N_SYM)])
        ax.set_xlabel("prossimo simbolo"); ax.set_ylabel("simbolo corrente")
        ax.set_title("Matrice dei legami appresa (ring buffer delle sequenze)\n"
                     "la diagonale sopra = A->B->C->D->E imparata")
        for i in range(N_SYM):
            for j in range(N_SYM):
                if M[i, j] > 0.05:
                    ax.text(j, i, "%.1f" % M[i, j], ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        out = "/mnt/user-data/outputs/sequence_learning.png"
        fig.savefig(out, dpi=130)
        print("chart -> %s" % out)
    except Exception as ex:
        print("[chart skipped: %s]" % ex)


if __name__ == "__main__":
    main()
