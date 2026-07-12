#!/usr/bin/env python3
# =============================================================================
#  neuron_st_v2.py
#  Aggiornamento del kernel ST. Novita' in SignalMgmt (DENDRITE):
#
#   1. L'accumulatore e' una CODA FINITA: occupazione limitata fra un FONDO
#      negativo (FLOOR) e un TETTO positivo (CEILING). I limiti finiti danno
#      la SATURAZIONE del meccanismo di accumulo:
#        - oltre il tetto: gli arrivi in eccesso si PERDONO (overflow) -> e' la
#          saturazione della depolarizzazione (driving force che si esaurisce);
#        - oltre il fondo: il DEBITO non scende oltre -> e' il limite
#          dell'iperpolarizzazione (potenziale di inversione del Cl-/K+).
#
#   2. DUE POOL per match di bitmask (mai peso negativo):
#        - ramo eccitatorio  -> ENQUEUE (deposita carica, sale)
#        - ramo inibitorio   -> WITHDRAW (preleva carica, scende nel debito)
#      Il peso resta un conteggio >= 0 di quanti; il SEGNO e' quale pool colpisci.
#
#   3. Il DEBITO ha memoria: un colpo inibitorio porta l'occupazione sotto zero;
#      il neurone deve prima "risalire il debito" prima di poter sparare di nuovo,
#      anche dopo che l'inibizione e' finita. E' l'iperpolarizzazione reale.
#
#  Il leak e' un SERVIZIO della coda che rilassa verso il riposo (0) da entrambi
#  i lati. Tutto intero. ActuatorCore (SOMA) invariato.
# =============================================================================

from enum import IntEnum


# ---- bitmask: i siti come maschere (ontologia VOS) ----
E_SITE = 0x0F     # (* sito eccitatorio *)
I_SITE = 0xF0     # (* sito inibitorio *)
TX_EXC = 0x03     # (* trasmettitore eccitatorio *)
TX_INH = 0x30     # (* trasmettitore inibitorio *)


class ActuatorStatus(IntEnum):
    Disabled = 0
    EnablePhase = 1
    Enabled = 2
    DisablePhase = 3


# =============================================================================
#  FUNCTION_BLOCK SignalMgmt   (* DENDRITE - coda finita E/I con debito *)
# =============================================================================
class SignalMgmt:

    def __init__(self, branchTransmitters):
        # branchTransmitters: lista di maschere, una per ramo
        # ---- VAR CONSTANT ----
        self.BRANCH_COUNT = len(branchTransmitters)
        self.QUANTUM_CHARGE = 100       # (* unita' di carica per quanto *)
        self.LEAK_RATE = 120            # (* servizio della coda: rilassa verso 0 *)
        self.THRESHOLD = 1000           # (* soglia che emette enableRequest *)
        self.CEILING = 1700             # (* TETTO della coda: satura qui *)
        self.FLOOR = -700               # (* FONDO della coda: debito massimo *)
        self.ELIG_LEAK = 1

        # maschere di sito del neurone post
        self.e_site_mask = E_SITE
        self.i_site_mask = I_SITE
        # maschera di trasmettitore per ramo
        self.branchTx = list(branchTransmitters)

        # ---- VAR_INPUT ----
        self.branchSpike = [False] * self.BRANCH_COUNT
        self.enabledSensor = False
        self.modulator = 0

        # ---- VAR_OUTPUT ----
        self.enableRequest = False

        # ---- VAR (ritenute) ----
        self.accumulator = 0            # (* occupazione della coda, fra FLOOR e CEILING *)
        self.overflow = 0               # (* carica persa per saturazione al tetto *)
        self.weight = [3] * self.BRANCH_COUNT
        self.elig = [0] * self.BRANCH_COUNT

    # ---- METHOD enqueue : deposito eccitatorio con saturazione al tetto ----
    def _enqueue(self, amount):
        self.accumulator = self.accumulator + amount
        if self.accumulator > self.CEILING:
            # (* coda piena: l'eccesso si perde -> saturazione dell'accumulo *)
            self.overflow = self.overflow + (self.accumulator - self.CEILING)
            self.accumulator = self.CEILING

    # ---- METHOD withdraw : prelievo inibitorio con saturazione al fondo ----
    def _withdraw(self, amount):
        self.accumulator = self.accumulator - amount
        if self.accumulator < self.FLOOR:
            # (* debito massimo raggiunto: non si scende oltre *)
            self.accumulator = self.FLOOR

    # ---- METHOD service : il leak rilassa la coda verso il riposo (0) ----
    def _service(self):
        if self.accumulator > 0:
            if self.accumulator > self.LEAK_RATE:
                self.accumulator = self.accumulator - self.LEAK_RATE
            else:
                self.accumulator = 0
        elif self.accumulator < 0:
            # (* recupero dal debito: risale verso 0 *)
            if self.accumulator < -self.LEAK_RATE:
                self.accumulator = self.accumulator + self.LEAK_RATE
            else:
                self.accumulator = 0

    def run(self):
        i = 0

        # (* 1. reset al riposo se il soma ha sparato (bAP/AHP) *)
        if self.enabledSensor:
            self.accumulator = 0

        # (* 2. servizio della coda (leak verso il riposo, da entrambi i lati) *)
        self._service()

        # (* 3. depositi/prelievi per match di bitmask (D->A, due pool) *)
        i = 0
        while i < self.BRANCH_COUNT:
            if self.branchSpike[i]:
                charge = self.weight[i] * self.QUANTUM_CHARGE
                if self.branchTx[i] & self.e_site_mask:
                    self._enqueue(charge)         # (* eccitatorio: sale *)
                elif self.branchTx[i] & self.i_site_mask:
                    self._withdraw(charge)        # (* inibitorio: scende nel debito *)
            i = i + 1

        # (* 4. eligibilita' per-ramo (coincidenza ramo + bAP) *)
        i = 0
        while i < self.BRANCH_COUNT:
            if self.elig[i] > self.ELIG_LEAK:
                self.elig[i] = self.elig[i] - self.ELIG_LEAK
            else:
                self.elig[i] = 0
            if self.branchSpike[i] and self.enabledSensor:
                self.elig[i] = self.elig[i] + 10
            i = i + 1

        # (* 5. commit del peso al modulatore *)
        if self.modulator != 0:
            i = 0
            while i < self.BRANCH_COUNT:
                step = self.elig[i] // 20
                if self.modulator > 0:
                    self.weight[i] = self.weight[i] + step
                else:
                    if self.weight[i] > step:
                        self.weight[i] = self.weight[i] - step
                    else:
                        self.weight[i] = 0
                i = i + 1

        # (* 6. soglia -> enableRequest (solo se l'occupazione RISALE sopra soglia) *)
        if self.accumulator >= self.THRESHOLD:
            self.enableRequest = True
        else:
            self.enableRequest = False


# =============================================================================
#  FUNCTION_BLOCK ActuatorCore   (* SOMA - invariato *)
# =============================================================================
class ActuatorCore:

    def __init__(self):
        self.SPIKE_WIDTH = 1
        self.REFRACTORY = 3
        self.enableRequest = False
        self.disableRequest = False
        self.enable = False
        self.disable = False
        self.enabledSensor = False
        self.status = ActuatorStatus.Disabled
        self.spikeTimer = 0
        self.refractoryTimer = 0

    def run(self):
        self.enable = False
        self.disable = False
        self.enabledSensor = False

        if self.disableRequest:
            self.status = ActuatorStatus.DisablePhase
            self.refractoryTimer = self.REFRACTORY

        if self.status == ActuatorStatus.Disabled:
            if self.enableRequest:
                self.status = ActuatorStatus.EnablePhase
                self.spikeTimer = self.SPIKE_WIDTH
        elif self.status == ActuatorStatus.EnablePhase:
            if self.spikeTimer > 0:
                self.spikeTimer = self.spikeTimer - 1
            if self.spikeTimer == 0:
                self.status = ActuatorStatus.Enabled
        elif self.status == ActuatorStatus.Enabled:
            self.enable = True
            self.enabledSensor = True
            self.status = ActuatorStatus.DisablePhase
            self.refractoryTimer = self.REFRACTORY
        elif self.status == ActuatorStatus.DisablePhase:
            self.disable = True
            if self.refractoryTimer > 0:
                self.refractoryTimer = self.refractoryTimer - 1
            if self.refractoryTimer == 0:
                self.status = ActuatorStatus.Disabled


# =============================================================================
#  FUNCTION_BLOCK Neuron   (* PERCEPTRON = SignalMgmt + ActuatorCore *)
# =============================================================================
class Neuron:

    def __init__(self, name, branchTransmitters):
        self.name = name
        self.dendrite = SignalMgmt(branchTransmitters)
        self.soma = ActuatorCore()

    def run(self):
        self.dendrite.run()
        self.soma.enableRequest = self.dendrite.enableRequest
        self.soma.run()
        self.dendrite.enabledSensor = self.soma.enabledSensor

    def reset_inputs(self):
        i = 0
        while i < self.dendrite.BRANCH_COUNT:
            self.dendrite.branchSpike[i] = False
            i = i + 1
        self.dendrite.modulator = 0
        self.soma.disableRequest = False


# =============================================================================
#  DEMO
# =============================================================================
def main():
    # ----- DEMO 1: debito inibitorio (iperpolarizzazione con memoria) -----
    # ramo 0 = eccitatorio (porta a sparare), ramo 1 = inibitorio (debito)
    n = Neuron("n", [TX_EXC, TX_INH])
    n.dendrite.weight[0] = 3     # eccitatorio
    n.dendrite.weight[1] = 5     # inibitorio (colpo forte)

    SCANS = 60
    acc1, sp1 = [], []
    inh_window = range(20, 24)   # raffica inibitoria
    c = 0
    while c < SCANS:
        n.reset_inputs()
        # eccitazione costante a partire dallo scan 2
        if c >= 2:
            n.dendrite.branchSpike[0] = True
        # inibizione nella finestra
        if c in inh_window:
            n.dendrite.branchSpike[1] = True
        n.run()
        acc1.append(n.dendrite.accumulator)
        if n.soma.enable:
            sp1.append(c)
        c = c + 1

    print("DEMO 1 - debito inibitorio")
    print("  spari del soma ai cicli: %s" % sp1)
    print("  finestra inibitoria: cicli %d-%d" % (inh_window.start, inh_window.stop - 1))
    print("  -> osserva il silenzio durante e DOPO l'inibizione (risalita del debito)")

    # ----- DEMO 2: saturazione al tetto (overflow) -----
    m = Neuron("m", [TX_EXC])
    m.dendrite.weight[0] = 9     # eccitazione molto forte
    acc2 = []
    c = 0
    while c < 25:
        m.reset_inputs()
        # NB: per vedere la saturazione disattivo il reset-da-spike alzando la soglia
        m.dendrite.THRESHOLD = 99999
        if c >= 2:
            m.dendrite.branchSpike[0] = True
        m.run()
        acc2.append(m.dendrite.accumulator)
        c = c + 1
    print("\nDEMO 2 - saturazione al tetto")
    print("  occupazione finale: %d (TETTO=%d)" % (acc2[-1], m.dendrite.CEILING))
    print("  carica persa per overflow: %d unita'" % m.dendrite.overflow)

    try:
        import os, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(acc1, color="#1f77b4", linewidth=2, label="occupazione coda")
        ax1.axhline(1000, color="k", linestyle="--", linewidth=1, label="THRESHOLD")
        ax1.axhline(0, color="grey", linewidth=0.8)
        ax1.axhline(-700, color="#d62728", linestyle=":", linewidth=1, label="FLOOR (debito max)")
        ax1.axhline(1700, color="#2ca02c", linestyle=":", linewidth=1, label="CEILING")
        ax1.axvspan(20, 23, color="#d62728", alpha=0.15, label="inibizione")
        for c in sp1:
            ax1.axvline(c, color="#2ca02c", linestyle=":", alpha=0.5)
        ax1.set_title("Debito inibitorio: l'inibizione spinge sotto zero,\n"
                      "il neurone tace finché non ha risalito il debito")
        ax1.set_xlabel("scan"); ax1.set_ylabel("occupazione (intero)")
        ax1.legend(fontsize=8, loc="lower right"); ax1.grid(alpha=0.3)

        ax2.plot(acc2, color="#ff7f0e", linewidth=2, label="occupazione coda")
        ax2.axhline(1700, color="#2ca02c", linestyle=":", linewidth=1.5, label="CEILING (saturazione)")
        ax2.set_title("Coda finita: l'accumulo SATURA al tetto,\n"
                      "gli arrivi in eccesso si perdono (overflow)")
        ax2.set_xlabel("scan"); ax2.set_ylabel("occupazione (intero)")
        ax2.legend(fontsize=9); ax2.grid(alpha=0.3)

        fig.tight_layout()
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        out = "/mnt/user-data/outputs/neuron_st_v2.png"
        fig.savefig(out, dpi=130)
        print("\nchart -> %s" % out)
    except Exception as ex:
        print("[chart skipped: %s]" % ex)


if __name__ == "__main__":
    main()
