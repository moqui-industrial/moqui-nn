#!/usr/bin/env python3
# =============================================================================
#  axon_synapse_st.py
#  Punti 2+3+4: il terzo sottosistema (ASSONE) + la SINAPSI dove il peso
#  NASCE dalla sincronia, non da un PWM dato.
#
#  Stile fedele a ST (FB con run() per scan, commenti VAR), ma matematica in
#  FLOAT come richiesto (quantizzazione a virgola fissa rimandata).
#
#  PUNTO 2 - Axon: ritardo PER-TERMINALE (non un ritardo unico). Un assone si
#            dirama a N terminali, ognuno col SUO ritardo di conduzione.
#  PUNTO 3 - Synapse: il peso NON e' un guadagno dato. E' la COVARIANZA
#            pre<->post mediata nella finestra inerziale:
#               cov = (pre_rate - <pre>) * (post_rate - <post>)
#            In fase (fluttuano insieme) -> cov>0 -> il peso sale.
#            Le medie mobili <.> sono leaky-average con tau = finestra inerziale.
#            Il peso e' la sincronia MISURATA e accumulata, poi modula gli
#            apporti futuri (apporto = peso * evento). Resta >=0 e satura
#            (il segno e' QUALE pool, l'inibizione e' un altro ramo).
#  PUNTO 4 - Lo sfasamento e' il segno NEGATIVO della stessa covarianza:
#            pre alto mentre post e' basso -> cov<0 -> il peso scende. Niente
#            ramo separato: una formula sola da' LTP e LTD.
#  GATE    - Sopra il livello base (non supervisionato) c'e' il terzo fattore:
#            la sincronia recente (elig) diventa peso DAVVERO quando arriva il
#            modulatore. Base propone, modulatore dispone.
# =============================================================================

from enum import IntEnum

E_SITE = 0x0F
I_SITE = 0xF0
TX_EXC = 0x03
TX_INH = 0x30


class Status(IntEnum):
    Disabled = 0
    EnablePhase = 1
    Enabled = 2
    DisablePhase = 3


# =============================================================================
#  FUNCTION_BLOCK Dendrite  (* coda finita; riceve APPORTI gia' pesati *)
# =============================================================================
class Dendrite:
    def __init__(self):
        self.LEAK_RATE = 120.0
        self.THRESHOLD = 1000.0
        self.CEILING = 1700.0
        self.FLOOR = -700.0
        # VAR_INPUT (impostati ogni scan dalla rete)
        self.i_exc = 0.0          # somma apporti eccitatori dalle sinapsi
        self.i_inh = 0.0          # somma apporti inibitori
        self.enabledSensor = False
        # VAR_OUTPUT
        self.enableRequest = False
        # VAR
        self.accumulator = 0.0

    def run(self):
        # 1. reset al riposo se il soma ha sparato (bAP/AHP)
        if self.enabledSensor:
            self.accumulator = 0.0
        # 2. servizio della coda (leak verso 0 da entrambi i lati)
        if self.accumulator > 0.0:
            self.accumulator = self.accumulator - self.LEAK_RATE
            if self.accumulator < 0.0:
                self.accumulator = 0.0
        elif self.accumulator < 0.0:
            self.accumulator = self.accumulator + self.LEAK_RATE
            if self.accumulator > 0.0:
                self.accumulator = 0.0
        # 3. deposito eccitatorio (satura al tetto)
        self.accumulator = self.accumulator + self.i_exc
        if self.accumulator > self.CEILING:
            self.accumulator = self.CEILING
        # 4. prelievo inibitorio (satura al fondo = debito)
        self.accumulator = self.accumulator - self.i_inh
        if self.accumulator < self.FLOOR:
            self.accumulator = self.FLOOR
        # 5. soglia -> enableRequest
        self.enableRequest = (self.accumulator >= self.THRESHOLD)


# =============================================================================
#  FUNCTION_BLOCK Soma  (* ActuatorCore: FSM one-shot + refrattario *)
# =============================================================================
class Soma:
    def __init__(self):
        self.SPIKE_WIDTH = 1
        self.REFRACTORY = 3
        self.enableRequest = False
        self.enable = False
        self.enabledSensor = False
        self.status = Status.Disabled
        self.spikeTimer = 0
        self.refractoryTimer = 0

    def run(self):
        self.enable = False
        self.enabledSensor = False
        if self.status == Status.Disabled:
            if self.enableRequest:
                self.status = Status.EnablePhase
                self.spikeTimer = self.SPIKE_WIDTH
        elif self.status == Status.EnablePhase:
            if self.spikeTimer > 0:
                self.spikeTimer = self.spikeTimer - 1
            if self.spikeTimer == 0:
                self.status = Status.Enabled
        elif self.status == Status.Enabled:
            self.enable = True
            self.enabledSensor = True
            self.status = Status.DisablePhase
            self.refractoryTimer = self.REFRACTORY
        elif self.status == Status.DisablePhase:
            if self.refractoryTimer > 0:
                self.refractoryTimer = self.refractoryTimer - 1
            if self.refractoryTimer == 0:
                self.status = Status.Disabled


# =============================================================================
#  FUNCTION_BLOCK Axon  (* PUNTO 2: ritardo PER-TERMINALE + fan-out *)
# =============================================================================
class Axon:
    def __init__(self, terminalDelays):
        # terminalDelays: lista, un ritardo (in scan) per ciascun terminale
        self.delays = list(terminalDelays)
        # un registro a scorrimento per terminale (FIFO di booleani)
        # lunghezza = delay esatto: lo spike emerge 'delay' scan dopo l'ingresso
        self.pipes = []
        i = 0
        while i < len(self.delays):
            d = self.delays[i]
            if d < 1:
                d = 1
            self.pipes.append([False] * d)
            i = i + 1
        # VAR_OUTPUT: evento in arrivo a ciascun terminale QUESTO scan
        self.out = [False] * len(self.delays)

    def run(self, spikeIn):
        # ogni terminale fa scorrere il proprio pipe del proprio ritardo
        t = 0
        while t < len(self.delays):
            pipe = self.pipes[t]
            # l'uscita e' la testa del pipe (l'evento entrato 'delay' scan fa)
            self.out[t] = pipe[0]
            # scorrimento
            k = 0
            while k < len(pipe) - 1:
                pipe[k] = pipe[k + 1]
                k = k + 1
            pipe[len(pipe) - 1] = spikeIn
            t = t + 1


# =============================================================================
#  FUNCTION_BLOCK Synapse  (* PUNTO 3+4: il peso E' la covarianza pre<->post *)
# =============================================================================
class Synapse:
    def __init__(self, transmitter=TX_EXC, weight=1.0):
        # ---- VAR CONSTANT ----
        self.QUANTUM = 150.0       # carica per unita' di peso
        self.TAU_RATE = 12.0       # finestra inerziale: leaky-avg dei rate
        self.TAU_MEAN = 80.0       # baseline lenta <pre>,<post>
        self.ETA_BASE = 25.0     # guadagno apprendimento NON supervisionato
        self.ETA_MOD = 6000.0     # guadagno del gate (terzo fattore)
        self.ELIG_LEAK = 0.92      # decadimento della sincronia recente
        self.W_MAX = 5.0           # saturazione del peso (finitismo)
        self.W_MIN = 0.0           # peso >=0: il segno e' QUALE pool
        self.transmitter = transmitter
        # ---- VAR (ritenute, float) ----
        self.weight = weight
        self.pre_rate = 0.0
        self.post_rate = 0.0
        self.pre_mean = 0.0
        self.post_mean = 0.0
        self.elig = 0.0
        self.last_cov = 0.0

    def _leaky(self, avg, x, tau):
        return avg + (x - avg) / tau

    def update(self, pre_event, post_spike, modulator):
        pe = 1.0 if pre_event else 0.0
        po = 1.0 if post_spike else 0.0
        # medie mobili nella finestra inerziale
        self.pre_rate = self._leaky(self.pre_rate, pe, self.TAU_RATE)
        self.post_rate = self._leaky(self.post_rate, po, self.TAU_RATE)
        # baseline lente <pre>,<post>
        self.pre_mean = self._leaky(self.pre_mean, self.pre_rate, self.TAU_MEAN)
        self.post_mean = self._leaky(self.post_mean, self.post_rate, self.TAU_MEAN)
        # COVARIANZA: in fase -> +, in sfasamento -> -  (punto 3 e punto 4)
        cov = (self.pre_rate - self.pre_mean) * (self.post_rate - self.post_mean)
        self.last_cov = cov
        # sincronia recente accumulata (la traccia)
        self.elig = self.elig * self.ELIG_LEAK + cov
        # LIVELLO BASE (non supervisionato): la sincronia muove il peso
        self.weight = self.weight + self.ETA_BASE * cov
        # GATE (terzo fattore): la sincronia recente si consolida col premio
        if modulator != 0.0:
            self.weight = self.weight + self.ETA_MOD * modulator * self.elig
        # saturazione del peso (>=0, <=W_MAX)
        if self.weight > self.W_MAX:
            self.weight = self.W_MAX
        if self.weight < self.W_MIN:
            self.weight = self.W_MIN

    def charge(self, pre_event):
        # apporto = peso * evento (il peso, nato dalla sincronia, modula l'apporto)
        if pre_event:
            return self.weight * self.QUANTUM
        return 0.0


# =============================================================================
#  Neurone e rete a 2 nodi
# =============================================================================
class Neuron:
    def __init__(self, name, terminalDelays):
        self.name = name
        self.dendrite = Dendrite()
        self.soma = Soma()
        self.axon = Axon(terminalDelays)


def force(dendrite):
    # impulso sopra-soglia per far sparare il soma a comando
    dendrite.i_exc = dendrite.THRESHOLD + 50.0


# =============================================================================
#  DEMO
# =============================================================================
def main():
    # --- dimostrazione PUNTO 2: ritardi per-terminale ---
    print("PUNTO 2 - Axon con ritardi per-terminale (fan-out):")
    demo_axon = Axon([2, 5, 9])
    arrivals = {0: [], 1: [], 2: []}
    c = 0
    while c < 14:
        spikeIn = (c == 1)   # un solo spike al ciclo 1
        demo_axon.run(spikeIn)
        t = 0
        while t < 3:
            if demo_axon.out[t]:
                arrivals[t].append(c)
            t = t + 1
        c = c + 1
    print("  spike emesso al ciclo 1 -> arriva ai terminali:")
    print("    term0 (delay 2): ciclo %s" % arrivals[0])
    print("    term1 (delay 5): ciclo %s" % arrivals[1])
    print("    term2 (delay 9): ciclo %s" % arrivals[2])

    # --- dimostrazione PUNTO 3+4: il peso nasce dalla sincronia ---
    pre = Neuron("pre", [1])      # un terminale, delay 1, verso post
    post = Neuron("post", [])
    syn = Synapse(transmitter=TX_EXC, weight=2.5)

    SCANS = 900
    w_trace = []
    cov_sign = []

    c = 0
    while c < SCANS:
        # default: nessun input forzato
        pre.dendrite.i_exc = 0.0; pre.dendrite.i_inh = 0.0
        post.dendrite.i_exc = 0.0; post.dendrite.i_inh = 0.0
        modulator = 0.0

        period = 14
        phase = c % period
        epoch = c // 300   # 0 = in fase, 1 = sfasato, 2 = in fase + premio

        # forzature dei due neuroni secondo l'epoca
        if epoch == 0:
            # IN FASE: pre poi post ravvicinati
            if phase == 0:
                force(pre.dendrite)
            if phase == 2:
                force(post.dendrite)
        elif epoch == 1:
            # SFASATO: pre e post in antifase (mezzo periodo di distanza)
            if phase == 0:
                force(pre.dendrite)
            if phase == 7:
                force(post.dendrite)
        else:
            # IN FASE + PREMIO
            if phase == 0:
                force(pre.dendrite)
            if phase == 2:
                force(post.dendrite)
            if phase == 5:
                modulator = +1.0    # ricompensa dopo la coincidenza

        # 1. assone del pre: propaga lo spike emesso lo scan precedente
        pre.axon.run(pre.soma.enable)
        pre_arrival = pre.axon.out[0]   # arrivo al terminale 0 (la sinapsi)

        # 2. la sinapsi consegna l'apporto pesato al dendrite del post
        ch = syn.charge(pre_arrival)
        if syn.transmitter & E_SITE:
            post.dendrite.i_exc = post.dendrite.i_exc + ch
        elif syn.transmitter & I_SITE:
            post.dendrite.i_inh = post.dendrite.i_inh + ch

        # 3. esegui dendrite + soma di entrambi
        pre.dendrite.run()
        pre.soma.enableRequest = pre.dendrite.enableRequest
        pre.soma.run()
        pre.dendrite.enabledSensor = pre.soma.enabledSensor

        post.dendrite.run()
        post.soma.enableRequest = post.dendrite.enableRequest
        post.soma.run()
        post.dendrite.enabledSensor = post.soma.enabledSensor

        # 4. aggiorna la sinapsi: covarianza pre-arrivo vs post-spike (+ gate)
        syn.update(pre_event=pre_arrival, post_spike=post.soma.enable, modulator=modulator)

        w_trace.append(syn.weight)
        c = c + 1

    print("\nPUNTO 3+4 - il peso nasce dalla sincronia:")
    print("  epoca 0 (IN FASE, no premio):   peso %.2f -> %.2f" % (w_trace[0], w_trace[299]))
    print("  epoca 1 (SFASATO, no premio):   peso %.2f -> %.2f" % (w_trace[300], w_trace[599]))
    print("  epoca 2 (IN FASE + premio):     peso %.2f -> %.2f" % (w_trace[600], w_trace[899]))

    try:
        import os, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(11, 5))
        ax.plot(w_trace, color="#1f77b4", linewidth=2)
        ax.axvspan(0, 300, color="#2ca02c", alpha=0.10)
        ax.axvspan(300, 600, color="#d62728", alpha=0.10)
        ax.axvspan(600, 900, color="#1f77b4", alpha=0.10)
        ax.text(150, ax.get_ylim()[1]*0.95, "IN FASE\n(sincronia)", ha="center", fontsize=9)
        ax.text(450, ax.get_ylim()[1]*0.95, "SFASATO\n(antifase)", ha="center", fontsize=9)
        ax.text(750, ax.get_ylim()[1]*0.95, "IN FASE\n+ PREMIO", ha="center", fontsize=9)
        ax.set_title("Il peso E' sincronia misurata: sale in fase, scende in antifase,\n"
                     "il modulatore consolida (covarianza pre<->post nella finestra inerziale)")
        ax.set_xlabel("scan"); ax.set_ylabel("peso sinaptico (float)")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        out = "/mnt/user-data/outputs/axon_synapse_st.png"
        fig.savefig(out, dpi=130)
        print("\nchart -> %s" % out)
    except Exception as ex:
        print("[chart skipped: %s]" % ex)


if __name__ == "__main__":
    main()
