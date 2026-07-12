#!/usr/bin/env python3
# =============================================================================
#  replay_consolidation.py   (FASE 3, punto 9)
#  Consolidamento offline tramite REPLAY del LogRingBuffer del PLC.
#  Allineato al FB reale: Push (sovrascrive il piu' vecchio quando pieno,
#  overflowCount), IncrementLast (repeatCount = compressione), Pop/Peek, Clear.
#
#  Idea: online l'esperienza viene solo REGISTRATA nel ring buffer (veloce,
#  come l'ippocampo). L'apprendimento dei pesi avviene OFFLINE, rileggendo il
#  buffer e ripresentando le transizioni (replay -> corteccia). Nessun dato
#  nuovo: la prestazione migliora solo consolidando cio' che e' gia' loggato.
#
#  Il repeatCount del tuo FB diventa REPLAY PRIORITIZZATO: le transizioni
#  ripetute (compresse con IncrementLast) vengono ripassate di piu'.
# =============================================================================

N_SYM = 6
LOG_MAX_SIZE = 12          # profondita' del ring buffer (come LOG_MAX_SIZE nel FB)
LR = 0.20


# --- LogRingBuffer: rispecchia il Function Block .st reale -------------------
class LogEvent:
    def __init__(self, prev, nxt):
        self.prevSym = prev          # payload: enumValue (transizione prev->nxt)
        self.nextSym = nxt
        self.repeatCount = 0         # come nel FB: ripetizioni compresse

    def __eq__(self, o):
        return isinstance(o, LogEvent) and self.prevSym == o.prevSym and self.nextSym == o.nextSym


class LogRingBuffer:
    def __init__(self, maxSize):
        self.LOG_MAX_SIZE = maxSize
        self.logBuffer = [None] * (maxSize + 1)   # 1-based come nel FB
        self.head = 1
        self.tail = 1
        self.logSize = 0
        self.overflowCount = 0

    def push(self, logEntry):
        # CASE LogCmd.Push
        self.logBuffer[self.tail] = logEntry
        if self.logSize >= self.LOG_MAX_SIZE:
            self.head = (self.head % self.LOG_MAX_SIZE) + 1     # sovrascrivi il piu' vecchio
            self.overflowCount += 1
        else:
            self.logSize += 1
        self.tail = (self.tail % self.LOG_MAX_SIZE) + 1

    def increment_last(self):
        # CASE LogCmd.IncrementLast: comprimi la ripetizione
        if self.logSize == 0:
            return False
        i = self.tail - 1
        if i < 1:
            i = self.LOG_MAX_SIZE
        if self.logBuffer[i].repeatCount < 65535:
            self.logBuffer[i].repeatCount += 1
        return True

    def last(self):
        i = self.tail - 1
        if i < 1:
            i = self.LOG_MAX_SIZE
        return self.logBuffer[i] if self.logSize > 0 else None

    def entries(self):
        # scorre da head, in ordine di inserimento (per il replay)
        out = []
        idx = self.head
        for _ in range(self.logSize):
            out.append(self.logBuffer[idx])
            idx = (idx % self.LOG_MAX_SIZE) + 1
        return out


# --- il "cortex": la matrice dei legami che si consolida col replay ---------
class Consolidator:
    def __init__(self):
        self.link = [[0.0] * N_SYM for _ in range(N_SYM)]

    def learn(self, prev, nxt, strength=1.0):
        self.link[prev][nxt] += LR * strength

    def predict(self, s):
        row = self.link[s]
        b = max(range(N_SYM), key=lambda j: row[j])
        return b if row[b] > 0 else -1


def log_experience(buf, seq):
    # ONLINE: registra soltanto (nessun apprendimento). Comprime le ripetizioni.
    for i in range(len(seq) - 1):
        ev = LogEvent(seq[i], seq[i + 1])
        last = buf.last()
        if last is not None and last == ev:
            buf.increment_last()          # stessa transizione -> repeatCount++
        else:
            buf.push(ev)


def replay(buf, cortex, cycles):
    # OFFLINE: rilegge il buffer e ripresenta le transizioni (replay).
    # repeatCount -> replay prioritizzato (le transizioni frequenti pesano di piu').
    for _ in range(cycles):
        for ev in buf.entries():
            cortex.learn(ev.prevSym, ev.nextSym, strength=1.0 + ev.repeatCount)


def accuracy(cortex, seq):
    ok = sum(1 for i in range(len(seq) - 1) if cortex.predict(seq[i]) == seq[i + 1])
    return ok / (len(seq) - 1)


def sym(i):
    return "ABCDEF"[i]


def main():
    print("Consolidamento offline via replay del LogRingBuffer (allineato al FB PLC)\n")
    SEQ = [0, 1, 2, 3, 4]
    print("Sequenza sperimentata: %s\n" % "->".join(sym(s) for s in SEQ))

    # ONLINE: vivi la sequenza poche volte, SOLO loggando nel ring buffer
    buf = LogRingBuffer(LOG_MAX_SIZE)
    for _ in range(2):
        log_experience(buf, SEQ)
    print("Dopo l'esperienza online (solo log, nessun apprendimento):")
    print("   ring buffer: logSize=%d, overflowCount=%d" % (buf.logSize, buf.overflowCount))

    print("\nVERIFICA - la prestazione migliora col REPLAY offline, senza nuovi dati:")
    print("   cicli di replay | accuratezza di predizione")
    cortex = Consolidator()
    print("        0        |      %.2f   (solo log: nulla di appreso)"
          % accuracy(cortex, SEQ))
    for cyc in [1, 2, 4, 8]:
        cortex = Consolidator()
        replay(buf, cortex, cyc)
        print("        %d        |      %.2f" % (cyc, accuracy(cortex, SEQ)))

    print("\n-> con 0 replay non si e' appreso nulla (online ha solo LOGGATO).")
    print("   Il consolidamento avviene rileggendo il buffer offline: la memoria")
    print("   passa dal ring buffer (ippocampo) ai pesi (corteccia) via replay.")
    print("   Nessun dato nuovo - solo ripasso di cio' che era gia' registrato.")

    # dimostrazione del replay prioritizzato via repeatCount
    print("\nVERIFICA 2 - replay prioritizzato (repeatCount): una transizione")
    print("   sperimentata piu' spesso si consolida prima.")
    buf2 = LogRingBuffer(LOG_MAX_SIZE)
    # A->B sperimentata 5 volte di fila (compressa), C->D una volta sola
    for _ in range(5):
        log_experience(buf2, [0, 1])       # A->B ripetuta
    log_experience(buf2, [2, 3])           # C->D una volta
    ab = buf2.entries()[0]
    print("   A->B repeatCount=%d,  C->D repeatCount=%d" % (ab.repeatCount, buf2.entries()[1].repeatCount))
    cortex2 = Consolidator()
    replay(buf2, cortex2, 1)
    print("   dopo 1 solo ciclo di replay: peso A->B=%.2f, peso C->D=%.2f"
          % (cortex2.link[0][1], cortex2.link[2][3]))
    print("   -> la transizione piu' vissuta e' gia' piu' forte (replay prioritizzato).")

    try:
        import os, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cycles = list(range(0, 11))
        accs = []
        for c in cycles:
            cx = Consolidator()
            if c > 0:
                replay(buf, cx, c)
            accs.append(accuracy(cx, SEQ))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(cycles, accs, "o-", color="#1f77b4")
        ax.set_title("Consolidamento offline: l'accuratezza sale coi cicli di replay\n"
                     "(nessun dato nuovo - solo ripasso del ring buffer)")
        ax.set_xlabel("cicli di replay offline"); ax.set_ylabel("accuratezza di predizione")
        ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        out = "/mnt/user-data/outputs/replay_consolidation.png"
        fig.savefig(out, dpi=130)
        print("\nchart -> %s" % out)
    except Exception as ex:
        print("[chart skipped: %s]" % ex)


if __name__ == "__main__":
    main()
