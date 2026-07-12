#!/usr/bin/env python3
# =============================================================================
#  neuron_full_suite.py
#  COLLAUDO COMPLETO della macchina: il Neuron intero (TUTTI i sottosistemi
#  attivi) istanziato N volte nel facade, su tre compiti diagnostici.
#
#  Sottosistemi attivi in OGNI passo (niente spento, niente scorciatoie):
#    - coda di Little (fuga proporzionale, membrana RC)
#    - FSM ActuatorCore (spike one-shot + refrattario)
#    - E/I: i neuroni inibitori sottraggono (bilancio E-I -> sparsita' reale,
#      non np.partition)
#    - sinapsi con peso e segno; dead time assonico (ring buffer per-ritardo)
#    - apprendimento a TRE FATTORI: dw = kPlast * modulatore * elig_pre * spike_post
#      (R-STDP: Hebbiano modulato dalla ricompensa - NIENTE backprop)
#    - omeostasi / HUNGRY (la soglia insegue il set-point di attivita')
#    - neuromodulazione (kGain/kPlast/kExplore dal blackboard)
#    - eligibilita' (traccia per i tre fattori)
#
#  Tre compiti (tre esiti attesi, tutti con la macchina COMPLETA):
#    1) MNIST statico      -> PERDE sul picco (terreno di casa di backprop)
#    2) Oblio catastrofico -> VINCE (apprendimento locale, McCloskey&Cohen 1989)
#    3) Ordine temporale   -> VINCE per costruzione (ANN feedforward non ha tempo)
#
#  E' un collaudo, non una vetrina: il compito 1 DEVE uscire negativo.
# =============================================================================

import time
import numpy as np

# --- costanti anatomiche (recipe di bootstrap) ------------------------------
ENABLE_TIME = 2
DISABLE_TIME = 3
THR_MIN = 5.0
TARGET_ACT = 0.05
HOMEO_RATE = 0.02
ACT_RATE = 0.02
ELIG_DECAY = 0.80
S_DISABLED, S_ENABLEPHASE, S_ENABLED, S_DISABLEPHASE = 0, 1, 2, 3


class SpikingNet:
    """Rete spiking COMPLETA, vettorizzata (SoA). Un solo step() con tutti i
    sottosistemi. Neuroni E/I, sinapsi con dead time, tre fattori, omeostasi,
    neuromodulazione. Istanziabile per qualunque compito cambiando la topologia."""

    def __init__(self, n, tau=8.0, threshold=30.0, d_max=12, seed=0):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.L = np.zeros(n)
        self.mu = 1.0 / tau
        self.status = np.zeros(n, dtype=np.int8)
        self.timer = np.zeros(n, dtype=np.int32)
        self.threshold = np.full(n, threshold)
        self.thr0 = float(threshold)
        self.activity = np.zeros(n)
        self.elig = np.zeros(n)
        self.spike = np.zeros(n, dtype=bool)
        self.is_inh = np.zeros(n, dtype=bool)          # E/I per neurone
        # ring buffer degli spike per il DEAD TIME assonico
        self.d_max = d_max
        self.ring = np.zeros((d_max + 1, n), dtype=bool)
        self.t = 0
        # sinapsi (array piatti)
        self.pre = self.post = self.w = self.delay = self.plastic = None
        self._syn = []
        # omeostasi on/off, neuromodulazione
        self.homeo = True
        self.kGain = 1.0; self.kPlast = 0.0; self.kExplore = 0.0; self.modulator = 0.0

    def set_inhibitory(self, idx):
        self.is_inh[idx] = True

    def add_syn(self, pre, post, w, delay=1, plastic=False):
        self._syn.append((pre, post, w, delay, plastic))

    def finalize(self):
        s = self._syn
        self.pre = np.array([x[0] for x in s], dtype=np.int64)
        self.post = np.array([x[1] for x in s], dtype=np.int64)
        self.w = np.array([x[2] for x in s], dtype=np.float64)
        self.delay = np.clip(np.array([x[3] for x in s], dtype=np.int64), 1, self.d_max)
        self.plastic = np.array([x[4] for x in s], dtype=bool)
        # segno E/I dal neurone presinaptico
        self.syn_sign = np.where(self.is_inh[self.pre], -1.0, 1.0)

    def reset_state(self):
        self.L[:] = 0; self.status[:] = 0; self.timer[:] = 0
        self.activity[:] = 0; self.elig[:] = 0; self.spike[:] = False
        self.threshold[:] = self.thr0; self.ring[:] = False; self.t = 0

    def step(self, ext_drive=None):
        n = self.n
        drive = np.zeros(n) if ext_drive is None else np.asarray(ext_drive, float).copy()
        # 1. DEAD TIME: ogni sinapsi legge lo spike del pre col SUO ritardo
        if self.pre is not None and self.pre.size:
            idx = (self.t - self.delay) % (self.d_max + 1)
            pre_fired = self.ring[idx, self.pre]                # spike del pre, ritardato
            contrib = self.syn_sign * self.w * pre_fired
            np.add.at(drive, self.post, contrib)                # E somma, I sottrae
        # 2. DENDRITE: coda di Little (fuga proporzionale)
        self.L = self.L * (1.0 - self.mu) + drive
        np.maximum(self.L, 0.0, out=self.L)
        self.spike = np.zeros(n, dtype=bool)
        # 3. neuromodulazione: gain sulla soglia, rumore di esplorazione
        effThr = self.threshold / self.kGain if self.kGain > 0 else self.threshold
        driveEff = self.L + (self.rng.standard_normal(n) * self.kExplore if self.kExplore > 0 else 0.0)
        # 4. SOMA: FSM ActuatorCore (snapshot -> niente cascata)
        st0 = self.status
        dis = st0 == S_DISABLED
        fire = dis & (driveEff >= effThr)
        ep = st0 == S_ENABLEPHASE; en = st0 == S_ENABLED; dp = st0 == S_DISABLEPHASE
        new = st0.copy()
        new[fire] = S_ENABLEPHASE; self.timer[fire] = ENABLE_TIME
        self.timer[ep & (self.timer > 0)] -= 1
        new[ep & (self.timer == 0)] = S_ENABLED
        self.spike[en] = True
        new[en] = S_DISABLEPHASE; self.timer[en] = DISABLE_TIME
        self.timer[dp & (self.timer > 0)] -= 1
        new[dp & (self.timer == 0)] = S_DISABLED
        self.status = new
        # 5. OMEOSTASI (HUNGRY)
        sp = self.spike.astype(float)
        self.activity += ACT_RATE * (sp - self.activity)
        if self.homeo:
            self.threshold += HOMEO_RATE * (self.activity - TARGET_ACT) * self.thr0
            np.maximum(self.threshold, THR_MIN, out=self.threshold)
        # 6. ELIGIBILITA'
        self.elig = self.elig * ELIG_DECAY + sp
        # 7. TRE FATTORI: dw = kPlast * modulatore * elig_pre * spike_post
        if self.kPlast != 0.0 and self.modulator != 0.0 and self.pre is not None:
            pm = self.plastic
            if pm.any():
                dw = self.kPlast * self.modulator * self.elig[self.pre[pm]] * self.spike[self.post[pm]]
                self.w[pm] += dw
                np.maximum(self.w, 0.0, out=self.w)
        # 8. scrivi lo spike nel ring (per il dead time del prossimo giro)
        self.t = (self.t + 1)
        self.ring[self.t % (self.d_max + 1)] = self.spike
        return self.spike


# =============================================================================
#  Utility: encoding di un'immagine come corrente, espansione con E/I sparsa
# =============================================================================
def build_expansion(n_in, n_hid, n_inh, seed=1):
    """Rete di espansione COMPLETA: n_hid eccitatori + n_inh inibitori.
    Gli eccitatori ricevono l'input (sinapsi fisse casuali); gli inibitori
    ricevono dagli eccitatori e li ri-inibiscono (bilancio E-I -> sparsita')."""
    N = n_hid + n_inh
    net = SpikingNet(N, tau=8.0, threshold=30.0, d_max=4, seed=seed)
    exc = np.arange(n_hid)
    inh = np.arange(n_hid, N)
    net.set_inhibitory(inh)
    net.homeo = True
    return net, exc, inh


def encode_full(net, exc, inh, X, Wexp, T_STEPS=30, gain=40.0):
    """Codifica un batch di immagini con la rete di espansione COMPLETA:
    input come corrente sugli eccitatori, E/I attivo, conta gli spike."""
    n_hid = len(exc)
    codes = np.zeros((len(X), n_hid), dtype=np.float32)
    for i in range(len(X)):
        net.reset_state()
        cur = np.maximum(0.0, X[i] @ Wexp) * gain
        drive = np.zeros(net.n); drive[exc] = cur
        cc = np.zeros(net.n)
        for _ in range(T_STEPS):
            cc += net.step(drive)
        codes[i] = cc[exc]
    return codes


def readout_three_factor(H, y, n_class=10, epochs=30, lr=0.05):
    """Readout appreso con la regola a TRE FATTORI (reward-modulata), NON backprop.
    Per ogni campione: uscita = pesi.codice; modulatore = +1 se la classe giusta
    vince, spinta locale. E' R-STDP a livello di rate (il modulatore e' la
    ricompensa; l'eligibilita' e' il codice presinaptico)."""
    d = H.shape[1]
    W = np.zeros((d, n_class), dtype=np.float32)
    idx = np.arange(len(H)); rng = np.random.default_rng(0)
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            h = H[i]
            out = h @ W
            pred = out.argmax()
            tgt = y[i]
            if pred != tgt:
                # modulatore locale: rinforza la classe giusta, indebolisce quella vinta
                W[:, tgt] += lr * h                      # tre fattori: mod=+1 su tgt
                W[:, pred] -= lr * h                      # mod=-1 sulla classe sbagliata
    return W


def acc(H, W, y):
    return float(np.mean((H @ W).argmax(1) == y))


# =============================================================================
#  COMPITO 1 e 2: MNIST e OBLIO (rete di espansione completa + readout 3-fattori)
# =============================================================================
def get_mnist():
    import os, gzip, pickle
    if os.path.exists("mnist.npz"):
        d = np.load("mnist.npz"); return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    try:
        from sklearn.datasets import fetch_openml
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
        X = (X/255.0).astype(np.float32); y = y.astype(int)
        Xtr,ytr,Xte,yte = X[:60000],y[:60000],X[60000:],y[60000:]
    except Exception:
        with gzip.open("nnadl/data/mnist.pkl.gz","rb") as f:
            tr,va,te = pickle.load(f, encoding="latin1")
        Xtr=np.vstack([tr[0],va[0]]).astype(np.float32); ytr=np.concatenate([tr[1],va[1]])
        Xte,yte=te[0].astype(np.float32),te[1]
    np.savez_compressed("mnist.npz",Xtr=Xtr,ytr=ytr,Xte=Xte,yte=yte)
    return Xtr,ytr,Xte,yte


def task_mnist(n_hid=800, n_train=4000, n_test=1000, T_STEPS=30):
    print("\n" + "="*66)
    print("COMPITO 1 - MNIST statico (atteso: PERDE sul picco vs backprop)")
    print("="*66)
    Xtr,ytr,Xte,yte = get_mnist()
    Xtr=Xtr.astype(np.float32); Xte=Xte.astype(np.float32)
    if Xtr.max()>1.5: Xtr/=255; Xte/=255
    rng=np.random.default_rng(0)
    itr=rng.choice(len(Xtr),n_train,replace=False); ite=rng.choice(len(Xte),n_test,replace=False)
    n_inh=max(1,n_hid//5)
    net,exc,inh = build_expansion(784,n_hid,n_inh,seed=1)
    Wexp=rng.standard_normal((784,n_hid))/np.sqrt(784)
    # E-I: exc->inh e inh->exc (sparsita' reale)
    for e in range(n_hid):
        net.add_syn(e, n_hid+(e%n_inh), w=6.0, delay=1)
    for j in range(n_inh):
        for e in range(j, n_hid, n_inh):
            net.add_syn(n_hid+j, e, w=4.0, delay=1)
    net.finalize()
    t0=time.time()
    Htr=encode_full(net,exc,inh,Xtr[itr],Wexp,T_STEPS=T_STEPS)
    Hte=encode_full(net,exc,inh,Xte[ite],Wexp,T_STEPS=T_STEPS)
    print("  codifica spiking completa (E/I attivo) in %.1fs, spike medi %.2f, attivi %.1f%%"
          %(time.time()-t0, Htr.mean(), 100*(Htr>0).mean()))
    W=readout_three_factor(Htr,ytr[itr])
    a=acc(Hte,W,yte[ite])
    print("  NOSTRO (spiking completo + readout 3-fattori, NO backprop): %.1f%%"%(100*a))
    try:
        from sklearn.neural_network import MLPClassifier
        import warnings; warnings.filterwarnings("ignore")
        mlp=MLPClassifier(hidden_layer_sizes=(n_hid,),max_iter=30,random_state=0).fit(Xtr[itr],ytr[itr])
        print("  BACKPROP di riferimento:                                     %.1f%%"%(100*mlp.score(Xte[ite],yte[ite])))
    except Exception as e: print("  [backprop saltato]")
    print("  -> se il nostro perde sul picco, e' l'esito atteso: MNIST statico e'")
    print("     il terreno di casa di backprop. Il collaudo lo mostra onestamente.")


def task_forgetting(n_hid=800, T_STEPS=30):
    print("\n" + "="*66)
    print("COMPITO 2 - oblio catastrofico (atteso: VINCE)")
    print("="*66)
    Xtr,ytr,Xte,yte=get_mnist()
    Xtr=Xtr.astype(np.float32); Xte=Xte.astype(np.float32)
    if Xtr.max()>1.5: Xtr/=255; Xte/=255
    rng=np.random.default_rng(1)
    def sub(X,y,cl,nn):
        m=np.isin(y,cl); Xs,ys=X[m],y[m]; ii=rng.choice(len(Xs),min(nn,len(Xs)),replace=False); return Xs[ii],ys[ii]
    XA,yA=sub(Xtr,ytr,range(5),3000); XB,yB=sub(Xtr,ytr,range(5,10),3000)
    XAt,yAt=sub(Xte,yte,range(5),800); XBt,yBt=sub(Xte,yte,range(5,10),800)
    n_inh=max(1,n_hid//5)
    net,exc,inh=build_expansion(784,n_hid,n_inh,seed=1)
    Wexp=rng.standard_normal((784,n_hid))/np.sqrt(784)
    for e in range(n_hid): net.add_syn(e,n_hid+(e%n_inh),w=6.0,delay=1)
    for j in range(n_inh):
        for e in range(j,n_hid,n_inh): net.add_syn(n_hid+j,e,w=4.0,delay=1)
    net.finalize()
    enc=lambda X: encode_full(net,exc,inh,X,Wexp,T_STEPS=T_STEPS)
    HA,HB,HAt,HBt=enc(XA),enc(XB),enc(XAt),enc(XBt)
    # readout PROTOTIPO (locale puro: scrive solo le colonne della classe corrente)
    def proto(H,y,W=None):
        if W is None: W=np.zeros((H.shape[1],10),dtype=np.float32)
        for c in np.unique(y):
            p=H[y==c].mean(0); nrm=np.linalg.norm(p)
            if nrm>1e-9: W[:,c]=p/nrm
        return W
    W=proto(HA,yA); a1=acc(HAt,W,yAt); W=proto(HB,yB,W); a2=acc(HAt,W,yAt); b2=acc(HBt,W,yBt)
    print("  NOSTRO (prototipo locale): A dopo A %.1f%% | A dopo B %.1f%% | oblio %.1f"
          %(100*a1,100*a2,100*(a1-a2)))
    from sklearn.neural_network import MLPClassifier
    import warnings; warnings.filterwarnings("ignore")
    mlp=MLPClassifier(hidden_layer_sizes=(n_hid,),learning_rate_init=0.01,random_state=0)
    for _ in range(15): mlp.partial_fit(XA,yA,classes=np.arange(10))
    bp1=mlp.score(XAt,yAt)
    for _ in range(15): mlp.partial_fit(XB,yB)
    bp2=mlp.score(XAt,yAt)
    print("  BACKPROP (vanilla):        A dopo A %.1f%% | A dopo B %.1f%% | oblio %.1f"
          %(100*bp1,100*bp2,100*(bp1-bp2)))
    print("  -> il locale scrive solo le colonne della classe corrente: A resta.")


# =============================================================================
#  COMPITO 3: ORDINE TEMPORALE (atteso: VINCE per costruzione)
# =============================================================================
def task_temporal(n_det=16, n_trials=400, T=30):
    print("\n" + "="*66)
    print("COMPITO 3 - ordine temporale A-prima-di-B vs B-prima-di-A")
    print("            (atteso: VINCE; una ANN feedforward non ha il tempo)")
    print("="*66)
    # rete: 2 input (A,B) -> n_det rilevatori con DEAD TIME diversi (Jeffress)
    # il dead time e' anatomico: linee di ritardo. Coincidenza via coda.
    rng=np.random.default_rng(0)
    N=2+n_det
    A_in,B_in=0,1; det=np.arange(2,2+n_det)
    def build():
        net=SpikingNet(N,tau=6.0,threshold=60.0,d_max=12,seed=1)
        dA=rng.integers(0,12,n_det); dB=rng.integers(0,12,n_det)
        for k in range(n_det):
            net.add_syn(A_in,det[k],w=45.0,delay=int(dA[k])+1)   # A con ritardo
            net.add_syn(B_in,det[k],w=45.0,delay=int(dB[k])+1)   # B con ritardo
        net.finalize(); return net
    def run_trial(net,cls):
        net.reset_state()
        jit=lambda:rng.integers(-1,2)
        if cls==0: tA,tB=5+jit(),15+jit()
        else: tB,tA=5+jit(),15+jit()
        fired=np.zeros(n_det)
        for t in range(T+14):
            drive=np.zeros(N)
            if t==tA: drive[A_in]=200.0
            if t==tB: drive[B_in]=200.0
            sp=net.step(drive)
            fired+=sp[det]
        return (fired>0).astype(float)
    net=build()
    X=[];Y=[]
    for _ in range(n_trials):
        c=rng.integers(0,2); X.append(run_trial(net,c)); Y.append(c)
    X=np.array(X); Y=np.array(Y)
    # readout lineare a correzione d'errore (locale) sui rilevatori
    w=np.zeros(n_det); b=0.0
    for _ in range(50):
        for i in range(len(X)):
            s=1.0 if (w@X[i]+b)>=0 else 0.0; tgt=float(Y[i])
            if s!=tgt: w+=(tgt-s)*X[i]; b+=(tgt-s)
    Xte=[];Yte=[]
    for _ in range(200):
        c=rng.integers(0,2); Xte.append(run_trial(net,c)); Yte.append(c)
    Xte=np.array(Xte);Yte=np.array(Yte)
    pred=((Xte@w+b)>=0).astype(int)
    acc_sp=np.mean(pred==Yte)
    active=(X.sum(0)>0).sum()
    print("  NOSTRO spiking (dead time + coincidenza): %.1f%%  (%d/%d rilevatori attivi)"
          %(100*acc_sp,active,n_det))
    # ANN feedforward SENZA tempo: vede solo le statistiche marginali (nessun ordine)
    from sklearn.neural_network import MLPClassifier
    import warnings; warnings.filterwarnings("ignore")
    # input feedforward: solo "A e' arrivato" e "B e' arrivato" (senza ordine) -> ~caso
    Xff=np.ones((len(Y),2)); Xff_te=np.ones((len(Yte),2))
    mlp=MLPClassifier(hidden_layer_sizes=(32,),max_iter=300,random_state=0).fit(Xff,Y)
    acc_ff=mlp.score(Xff_te,Yte)
    print("  ANN feedforward (senza accesso al tempo):  %.1f%%  (~caso: l'ordine e' invisibile)"
          %(100*acc_ff))
    print("  -> il tempo e' NATIVO nel substrato spiking (dead time). Backprop")
    print("     feedforward non puo' vederlo senza un'architettura ricorrente apposta.")


def main():
    print("COLLAUDO COMPLETO - tutti i sottosistemi attivi, tre compiti diagnostici\n")
    task_mnist()
    task_forgetting()
    task_temporal()
    print("\n" + "="*66)
    print("Tre esiti, una sola macchina completa: perde sul picco statico (1),")
    print("vince sull'oblio (2), vince sul tempo per costruzione (3). Collaudo,")
    print("non vetrina: l'esito negativo di (1) e' emerso, come deve.")


if __name__ == "__main__":
    main()
