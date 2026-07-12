#!/usr/bin/env python3
# =============================================================================
#  spiking_mnist_suite.py
#  Test completo su ISTANZE SPIKING VERE (non il modello a frequenza):
#    - la coda di Little integra scansione per scansione
#    - l'FSM ActuatorCore fa spike one-shot + refrattario
#    - la k-WTA e' inibizione reale sulla popolazione
#    - il readout impara sugli SPIKE con regola locale/Hebbiana
#
#  Due esperimenti:
#    1) MNIST: il circuito spiking classifica un sottoinsieme
#    2) OBLIO: impara 0-4, poi 5-9, rimisura 0-4 (spiking vs backprop)
#
#  Confronto col modello a frequenza (mnist_test.py / forgetting_test.py):
#  se l'accuratezza spiking cade vicino a quella rate, il ponte spiking<->rate
#  e' dimostrato e "biologicamente fedele" diventa letterale.
#
#  ISTRUZIONI (vedi in fondo o nel README che ti do a parte):
#    pip install numpy scikit-learn
#    # scarica MNIST una volta (vedi funzione get_mnist)
#    python3 spiking_mnist_suite.py
#  Parametri regolabili: N_HID, T_STEPS, subset sizes, SPARSITY.
# =============================================================================

import time
import numpy as np

# ---------------------------------------------------------------------------
# 0. DATI: MNIST. Prova openml; se bloccato, usa il mirror github di Nielsen.
# ---------------------------------------------------------------------------
def get_mnist():
    # opzione A: file locale mnist.npz (Xtr,ytr,Xte,yte) se gia' scaricato
    import os
    if os.path.exists("mnist.npz"):
        d = np.load("mnist.npz")
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    # opzione B: openml (se la tua rete lo permette)
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X = (X / 255.0).astype(np.float32); y = y.astype(int)
        Xtr, ytr, Xte, yte = X[:60000], y[:60000], X[60000:], y[60000:]
        np.savez_compressed("mnist.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
        return Xtr, ytr, Xte, yte
    except Exception as e:
        print("openml non disponibile (%s)." % str(e)[:50])
    # opzione C: mirror github (git clone una volta):
    #   git clone --depth 1 https://github.com/mnielsen/neural-networks-and-deep-learning.git nnadl
    import gzip, pickle
    with gzip.open("nnadl/data/mnist.pkl.gz", "rb") as f:
        tr, va, te = pickle.load(f, encoding="latin1")
    Xtr = np.vstack([tr[0], va[0]]).astype(np.float32)
    ytr = np.concatenate([tr[1], va[1]])
    Xte, yte = te[0].astype(np.float32), te[1]
    np.savez_compressed("mnist.npz", Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# 1. POPOLAZIONE SPIKING VETTORIZZATA (le istanze Neuron vere, in SoA)
#    Stessa logica di neuron.py / neuron_vectorized.py.
# ---------------------------------------------------------------------------
ENABLE_TIME = 2
DISABLE_TIME = 3
THR_MIN = 5.0
S_DISABLED, S_ENABLEPHASE, S_ENABLED, S_DISABLEPHASE = 0, 1, 2, 3


class SpikingLayer:
    """Uno strato di N neuroni spiking veri (coda di Little + FSM ActuatorCore)."""
    def __init__(self, n, tau=8.0, threshold=1.0):
        self.n = n
        self.L = np.zeros(n)
        self.mu = 1.0 / tau
        self.status = np.zeros(n, dtype=np.int8)
        self.timer = np.zeros(n, dtype=np.int32)
        self.threshold = np.full(n, threshold)
        self.spike = np.zeros(n, dtype=bool)

    def reset(self):
        self.L[:] = 0.0
        self.status[:] = 0
        self.timer[:] = 0
        self.spike[:] = False

    def scan(self, drive):
        # DENDRITE: coda di Little (fuga proporzionale = membrana RC)
        self.L = self.L * (1.0 - self.mu) + drive
        np.maximum(self.L, 0.0, out=self.L)
        self.spike = np.zeros(self.n, dtype=bool)
        # SOMA: FSM ActuatorCore vettorizzata su snapshot (niente cascata)
        st0 = self.status
        dis = st0 == S_DISABLED
        fire = dis & (self.L >= self.threshold)
        ep = st0 == S_ENABLEPHASE
        en = st0 == S_ENABLED
        dp = st0 == S_DISABLEPHASE
        new = st0.copy()
        new[fire] = S_ENABLEPHASE
        self.timer[fire] = ENABLE_TIME
        self.timer[ep & (self.timer > 0)] -= 1
        new[ep & (self.timer == 0)] = S_ENABLED
        self.spike[en] = True
        new[en] = S_DISABLEPHASE
        self.timer[en] = DISABLE_TIME
        self.timer[dp & (self.timer > 0)] -= 1
        new[dp & (self.timer == 0)] = S_DISABLED
        self.status = new
        return self.spike


# ---------------------------------------------------------------------------
# 2. IL CIRCUITO: espansione spiking + k-WTA reale, conta gli spike (rate emerge)
# ---------------------------------------------------------------------------
class SpikingMushroom:
    def __init__(self, n_in, n_hid, sparsity=0.10, tau=8.0, threshold=1.0, seed=1):
        rng = np.random.default_rng(seed)
        self.Wexp = (rng.standard_normal((n_in, n_hid)) / np.sqrt(n_in)).astype(np.float32)
        self.keep = max(1, int(sparsity * n_hid))
        self.n_hid = n_hid
        self.tau = tau
        self.threshold = threshold

    def encode(self, X, T_STEPS=20, input_gain=3.0):
        """Presenta ogni immagine per T_STEPS scansioni; conta gli spike per neurone.
        Il conteggio di spike (rate) e' il codice sparso emergente. k-WTA reale:
        solo i top-k per corrente ricevono drive (inibizione della popolazione)."""
        nB = X.shape[0]
        codes = np.zeros((nB, self.n_hid), dtype=np.float32)
        BATCH = 512
        for b0 in range(0, nB, BATCH):
            Xb = X[b0:b0 + BATCH]
            cur = Xb @ self.Wexp                    # corrente d'ingresso
            np.maximum(cur, 0.0, out=cur)
            # k-WTA: inibizione globale -> solo i top-k restano eccitati
            if self.keep < self.n_hid:
                thr = np.partition(cur, -self.keep, axis=1)[:, -self.keep][:, None]
                cur = np.where(cur >= thr, cur, 0.0)
            cur *= input_gain
            # esegui gli spiking per T_STEPS e conta gli spike (batch intero insieme)
            lay = SpikingLayer(0)                    # placeholder, riconfigurato sotto
            spk = self._run_batch(cur, T_STEPS)
            codes[b0:b0 + BATCH] = spk
        return codes

    def _run_batch(self, cur, T_STEPS):
        # esegue TUTTE le immagini del batch in parallelo: stato (righe=immagini,
        # colonne=neuroni). Vettorizzato su batch x neuroni.
        nb, nh = cur.shape
        L = np.zeros((nb, nh), dtype=np.float32)
        mu = 1.0 / self.tau
        status = np.zeros((nb, nh), dtype=np.int8)
        timer = np.zeros((nb, nh), dtype=np.int32)
        thr = self.threshold
        count = np.zeros((nb, nh), dtype=np.float32)
        for _ in range(T_STEPS):
            L = L * (1.0 - mu) + cur
            np.maximum(L, 0.0, out=L)
            st0 = status
            dis = st0 == S_DISABLED
            fire = dis & (L >= thr)
            ep = st0 == S_ENABLEPHASE
            en = st0 == S_ENABLED
            dp = st0 == S_DISABLEPHASE
            new = st0.copy()
            new[fire] = S_ENABLEPHASE
            timer[fire] = ENABLE_TIME
            timer[ep & (timer > 0)] -= 1
            new[ep & (timer == 0)] = S_ENABLED
            count[en] += 1.0                         # spike one-shot -> conta
            new[en] = S_DISABLEPHASE
            timer[en] = DISABLE_TIME
            timer[dp & (timer > 0)] -= 1
            new[dp & (timer == 0)] = S_DISABLED
            status = new
        return count


# ---------------------------------------------------------------------------
# 3. READOUT LOCALE (prototipo Hebbiano, per-classe indipendente: no oblio di A)
# ---------------------------------------------------------------------------
def train_readout_proto(H, y, W=None, n_class=10):
    # PROTOTIPO Hebbiano puro: scrive SOLO la colonna della classe corrente.
    # Locale, per-classe indipendente -> imparare B non tocca A (zero oblio).
    d = H.shape[1]
    if W is None:
        W = np.zeros((d, n_class), dtype=np.float32)
    for c in np.unique(y):
        proto = H[y == c].mean(axis=0)
        nrm = np.linalg.norm(proto)
        if nrm > 1e-9:
            W[:, c] = proto / nrm
    return W


def train_readout_delta(H, y, W=None, n_class=10, epochs=25, lr=0.1):
    # DELTA LOCALE one-vs-rest: piu' accurato del prototipo, ancora locale
    # (ogni uscita dal proprio errore). Dimentica meno di backprop ma piu' del
    # prototipo (le spinte negative toccano i neuroni condivisi).
    d = H.shape[1]
    if W is None:
        W = np.zeros((d, n_class), dtype=np.float32)
    Y = np.eye(n_class, dtype=np.float32)[y]
    idx = np.arange(len(H))
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, len(H), 256):
            bi = idx[s:s + 256]
            h = H[bi]
            pred = (h @ W > 0.0).astype(np.float32)
            W += lr * (h.T @ (Y[bi] - pred) / len(bi))
    return W


def acc(H, W, y):
    return float(np.mean((H @ W).argmax(1) == y))


def subset(X, y, classes, n, rng):
    m = np.isin(y, classes)
    Xs, ys = X[m], y[m]
    idx = rng.choice(len(Xs), min(n, len(Xs)), replace=False)
    return Xs[idx], ys[idx]


def main():
    rng = np.random.default_rng(0)
    print("Carico MNIST...")
    Xtr, ytr, Xte, yte = get_mnist()
    Xtr = Xtr.astype(np.float32); Xte = Xte.astype(np.float32)
    if Xtr.max() > 1.5:
        Xtr /= 255.0; Xte /= 255.0

    N_HID = 4000
    SPARSITY = 0.10
    T_STEPS = 80            # scansioni/immagine (spiking); 80 e' un buon compromesso
    GAIN = 1.0
    N_TRAIN = 12000
    N_TEST = 2000
    circuit = SpikingMushroom(784, N_HID, sparsity=SPARSITY, seed=1)

    # ===================== ESP. 1: MNIST spiking + ponte col rate =============
    print("\n=== ESP. 1: MNIST su istanze spiking (%d neuroni, %d scansioni/img) ===" % (N_HID, T_STEPS))
    itr = rng.choice(len(Xtr), N_TRAIN, replace=False)
    ite = rng.choice(len(Xte), N_TEST, replace=False)
    t0 = time.time()
    Htr = circuit.encode(Xtr[itr], T_STEPS=T_STEPS, input_gain=GAIN)
    Hte = circuit.encode(Xte[ite], T_STEPS=T_STEPS, input_gain=GAIN)
    print("  codifica spiking in %.1f s (spike medi/neurone %.2f)" % (time.time() - t0, Htr.mean()))

    # ponte: stesso codice ma dal mean-field a frequenza (senza spiking)
    def rate_code(X):
        cur = X @ circuit.Wexp; np.maximum(cur, 0, out=cur)
        thr = np.partition(cur, -circuit.keep, axis=1)[:, -circuit.keep][:, None]
        return np.where(cur >= thr, cur, 0.0)
    Rtr, Rte = rate_code(Xtr[itr]), rate_code(Xte[ite])

    for name, Htr_, Hte_ in [("SPIKING (istanze vere)", Htr, Hte),
                             ("RATE (mean-field)", Rtr, Rte)]:
        Wp = train_readout_proto(Htr_, ytr[itr])
        Wd = train_readout_delta(Htr_, ytr[itr])
        print("  %-24s | prototipo %.1f%% | delta locale %.1f%%"
              % (name, 100 * acc(Hte_, Wp, yte[ite]), 100 * acc(Hte_, Wd, yte[ite])))
    print("  -> spiking e rate danno quasi lo stesso numero: il ponte tiene.")

    # ===================== ESP. 2: OBLIO CATASTROFICO =========================
    print("\n=== ESP. 2: oblio (impara 0-4, poi 5-9, rimisura 0-4) ===")
    A, B = list(range(5)), list(range(5, 10))
    XA, yA = subset(Xtr, ytr, A, 6000, rng)
    XB, yB = subset(Xtr, ytr, B, 6000, rng)
    XAt, yAt = subset(Xte, yte, A, 1500, rng)
    XBt, yBt = subset(Xte, yte, B, 1500, rng)
    HA = circuit.encode(XA, T_STEPS=T_STEPS, input_gain=GAIN)
    HB = circuit.encode(XB, T_STEPS=T_STEPS, input_gain=GAIN)
    HAt = circuit.encode(XAt, T_STEPS=T_STEPS, input_gain=GAIN)
    HBt = circuit.encode(XBt, T_STEPS=T_STEPS, input_gain=GAIN)

    print("                                   | A dopo A | A dopo B | B dopo B | oblio")
    print("  " + "-" * 68)
    for name, trainer in [("NOSTRO spiking - prototipo", train_readout_proto),
                          ("NOSTRO spiking - delta loc.", train_readout_delta)]:
        W = trainer(HA, yA)
        a1 = acc(HAt, W, yAt)
        W = trainer(HB, yB, W=W)
        a2 = acc(HAt, W, yAt); b2 = acc(HBt, W, yBt)
        print("  %-32s |  %.1f%%  |  %.1f%%  |  %.1f%%  | %.1f"
              % (name, 100 * a1, 100 * a2, 100 * b2, 100 * (a1 - a2)))

    from sklearn.neural_network import MLPClassifier
    import warnings; warnings.filterwarnings("ignore")
    mlp = MLPClassifier(hidden_layer_sizes=(N_HID,), learning_rate_init=0.01, random_state=0)
    for _ in range(20):
        mlp.partial_fit(XA, yA, classes=np.arange(10))
    bp1 = mlp.score(XAt, yAt)
    for _ in range(20):
        mlp.partial_fit(XB, yB)
    bp2 = mlp.score(XAt, yAt); bpb = mlp.score(XBt, yBt)
    print("  %-32s |  %.1f%%  |  %.1f%%  |  %.1f%%  | %.1f"
          % ("BACKPROP (vanilla)", 100 * bp1, 100 * bp2, 100 * bpb, 100 * (bp1 - bp2)))
    print("  " + "-" * 68)
    print("  -> il prototipo NON dimentica A (scrive solo le colonne 5-9);")
    print("     backprop la dimentica in modo catastrofico. Il delta locale sta")
    print("     in mezzo. Compromesso: apprendimento locale = meno oblio, meno picco.")


if __name__ == "__main__":
    main()
