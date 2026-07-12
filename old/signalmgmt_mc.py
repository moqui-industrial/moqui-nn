#!/usr/bin/env python3
# =============================================================================
#  signalmgmt_mc.py
#  Le funzioni MC_ tradotte in BITFIELD sulla stessa macchina tag&maschera di
#  SignalMgmt. Nessun function block nuovo. Minimale: 2 bit di stato + 1 bit
#  nel tag. Ognuno con il suo referente fisiologico.
#
#   MC_TouchProbe          -> PROBED  (bitmask per-ramo): quali master sono
#                             arrivati DENTRO la finestra prima del mio spike.
#                             Fisiologia: rilevatore di coincidenza NMDA / STDP.
#   MC_ReadActual+ReadSet  -> HUNGRY  (1 bit per-neurone): actual_rate < set_rate.
#                             I due "read" collassano nel SEGNO della differenza.
#                             Fisiologia: plasticita' omeostatica (Turrigiano).
#                             Se affamato abbassa la soglia -> cerca master.
#   MC_CamIn (drv/mod)      -> DRV    (1 bit nel tag): driver forte vs modulator
#                             debole. Fisiologia: driver/modulator (Sherman&Guillery).
#                             L'ingaggio in se' e' GIA' il match tag&site esistente.
# =============================================================================

# --- tag a 8 bit: nibble E (0x0F) e nibble I (0xF0), gia' esistenti ---
E_SITE = 0x0F
I_SITE = 0xF0
TX_EXC = 0x03
TX_INH = 0x30
DRIVER = 0x44      # bit 'driver' in ciascun nibble (0x04 lato E, 0x40 lato I)

# --- parametri fisici (pochi) ---
WINDOW = 4         # scan: la finestra d'inerzia (il cuscinetto della sincronia)
MOD_GAIN = 0.25    # il modulator conta poco; il driver pieno
SET_RATE = 0.10    # set-point omeostatico di attivita'
HUNGRY_DROP = 350.0  # di quanto cala la soglia quando affamato (cerca master)
RATE_TAU = 20.0


class SignalMgmtMC:
    """Dendrite a coda finita + i tre bit MC_. Stessa coda, piu' bitfield."""

    def __init__(self, n_masters):
        self.LEAK_RATE = 120.0
        self.THRESHOLD = 1000.0
        self.CEILING = 1700.0
        self.FLOOR = -700.0
        self.accumulator = 0.0
        self.enableRequest = False
        self.enabledSensor = False
        # --- stato bitfield ---
        self.n_masters = n_masters
        self.probe_ring = [0] * WINDOW   # maschere di arrivo degli ultimi WINDOW scan
        self.probed = 0                  # OR del ring = master in finestra (TouchProbe)
        self.actual_rate = 0.0
        self.hungry = False              # ReadActual < ReadSet
        # accumulo dello scan corrente
        self._arr = 0
        self._i_exc = 0.0
        self._i_inh = 0.0

    def eff_threshold(self):
        # HUNGRY abbassa la soglia: il neurone affamato cerca master (omeostasi)
        return self.THRESHOLD - (HUNGRY_DROP if self.hungry else 0.0)

    def deposit(self, master_bit, charge, tag):
        # MC_CamIn: il ramo ingaggia il master (il match tag&site e' gia' avvenuto
        #           a monte; qui applichiamo driver vs modulator).
        g = 1.0 if (tag & DRIVER) else MOD_GAIN
        if tag & E_SITE:
            self._i_exc += charge * g
        else:
            self._i_inh += charge * g
        # MC_TouchProbe: segna che QUESTO master e' arrivato in questo scan
        self._arr |= (1 << master_bit)

    def run(self):
        # 1. bAP: il mio spike azzera la coda (AHP)
        if self.enabledSensor:
            self.accumulator = 0.0
        # 2. leak = la finestra d'inerzia (drena verso 0 da entrambi i lati)
        if self.accumulator > 0.0:
            self.accumulator = max(0.0, self.accumulator - self.LEAK_RATE)
        elif self.accumulator < 0.0:
            self.accumulator = min(0.0, self.accumulator + self.LEAK_RATE)
        # 3. deposita exc/inh dello scan (driver pieni, modulator scalati)
        self.accumulator = min(self.CEILING, self.accumulator + self._i_exc)
        self.accumulator = max(self.FLOOR, self.accumulator - self._i_inh)
        # 4. MC_TouchProbe: aggiorna la finestra dei master arrivati
        self.probe_ring.pop(0)
        self.probe_ring.append(self._arr)
        self.probed = 0
        for m in self.probe_ring:
            self.probed |= m
        # 5. soglia (abbassata se HUNGRY) -> enableRequest
        fired = self.accumulator >= self.eff_threshold()
        self.enableRequest = fired
        # 6. MC_ReadActual vs ReadSet -> HUNGRY (per il prossimo scan)
        self.actual_rate += ((1.0 if fired else 0.0) - self.actual_rate) / RATE_TAU
        self.hungry = (self.actual_rate < SET_RATE)
        # reset accumulo scan
        self._arr = 0
        self._i_exc = 0.0
        self._i_inh = 0.0

    def probed_masters(self):
        return [m for m in range(self.n_masters) if (self.probed >> m) & 1]


def b(x, n):
    return format(x, "0%db" % n)


# =============================================================================
#  DEMO dei tre bit
# =============================================================================
def demo_probed():
    print("=" * 70)
    print("PROBED (MC_TouchProbe): quali master sono in finestra (WINDOW=%d)" % WINDOW)
    print("=" * 70)
    sm = SignalMgmtMC(n_masters=3)
    # master 0 arriva allo scan 1, master 1 allo scan 2, master 2 allo scan 6
    arrivi = {1: [0], 2: [1], 6: [2]}
    print("scan | arrivi | PROBED (bit m=1 se master m e' in finestra)")
    for s in range(10):
        for m in arrivi.get(s, []):
            sm.deposit(master_bit=m, charge=200.0, tag=TX_EXC)
        sm.run()
        print("  %2d |  %-6s| %s   master in finestra: %s"
              % (s, str(arrivi.get(s, [])), b(sm.probed, 3), sm.probed_masters()))
    print("-> ogni master resta nella maschera %d scan, poi esce (la finestra)\n" % WINDOW)


def demo_hungry():
    print("=" * 70)
    print("HUNGRY (MC_ReadActual<ReadSet): affamato -> abbassa la soglia, cerca master")
    print("=" * 70)
    sm = SignalMgmtMC(n_masters=1)
    print("fase 1: digiuno (nessun input)")
    print("scan | rate  | HUNGRY | soglia_eff")
    for s in range(6):
        sm.run()
        print("  %2d | %.3f |  %s  | %.0f" % (s, sm.actual_rate, sm.hungry, sm.eff_threshold()))
    print("fase 2: arriva un master forte ogni scan")
    for s in range(6, 16):
        sm.deposit(0, 900.0, TX_EXC | DRIVER)
        sm.run()
        print("  %2d | %.3f |  %s  | %.0f%s"
              % (s, sm.actual_rate, sm.hungry, sm.eff_threshold(),
                 "  <- sazio: soglia torna alta" if not sm.hungry else ""))
    print("-> affamato la soglia cala (aggancio facile); sazio risale (smette di cercare)\n")


def demo_driver_vs_modulator():
    print("=" * 70)
    print("DRV (MC_CamIn): 1 driver forte spara; molti modulator deboli no")
    print("=" * 70)
    # molti modulator (un solo scan, neurone sazio cosi' la soglia resta piena)
    sm = SignalMgmtMC(n_masters=8)
    sm.actual_rate = 1.0      # non affamato: soglia piena 1000
    for m in range(8):
        sm.deposit(m, 300.0, TX_EXC)            # 8 modulator (gain 0.25 -> 75 cad.)
    sm.run()
    print("8 modulator insieme -> accumulatore %.0f, spara=%s (soglia %.0f)"
          % (sm.accumulator, sm.enableRequest, sm.eff_threshold()))
    # un driver
    sm2 = SignalMgmtMC(n_masters=1)
    sm2.actual_rate = 1.0
    sm2.deposit(0, 1100.0, TX_EXC | DRIVER)     # 1 driver (gain 1.0)
    sm2.run()
    print("1 driver da solo      -> accumulatore %.0f, spara=%s (soglia %.0f)"
          % (sm2.accumulator, sm2.enableRequest, sm2.eff_threshold()))
    print("-> il driver determina; i modulator regolano il guadagno (Sherman&Guillery)\n")


if __name__ == "__main__":
    demo_probed()
    demo_hungry()
    demo_driver_vs_modulator()
