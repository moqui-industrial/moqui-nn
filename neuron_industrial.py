#!/usr/bin/env python3
"""
neuron_industrial.py

Nuovo framework per Reti Neurali Artificiali Spiking (SNN) basato
sui design pattern dell'automazione industriale (IEC 61131-3) e 
teoria delle code di eventi discreti.

- SOMA: ActuatorCore (FSM)
- DENDRITE: SignalMgmtMC (Coda di Little, FLOOR/CEILING)
- ASSONE: AxonDelay (Dead time)
- SINAPSI: Synapse (Covarianza / Camma elettronica)
- MEMORIA: LogRingBuffer (Replay offline)
"""

from enum import IntEnum

# --- Costanti e Bitmask ---
E_SITE = 0x0F     # sito eccitatorio
I_SITE = 0xF0     # sito inibitorio
TX_EXC = 0x03     # trasmettitore eccitatorio
TX_INH = 0x30     # trasmettitore inibitorio
TX_SHUNT = 0xC0   # trasmettitore inibitorio (divisivo/shunting)
DRIVER = 0x100    # bit 'driver' (evita collisioni con il byte 0-7)

class ActuatorStatus(IntEnum):
    Disabled = 0
    EnablePhase = 1
    Enabled = 2
    DisablePhase = 3


# =============================================================================
#  1. Blackboard (Control Plane Globale / Locus Coeruleus)
# =============================================================================
class Blackboard:
    """Variabili condivise per la neuromodulazione (Task Lento -> Task Veloce)"""
    def __init__(self):
        self.kGain = 1.0          # Eccitabilita'
        self.kPlasticity = 0.0    # Tasso di apprendimento
        self.kExplore = 0.0       # Rumore
        self.regimeName = "esegue"


# =============================================================================
#  2. LogRingBuffer (Ippocampo / Traccia Eligibilita')
# =============================================================================
class LogEvent:
    def __init__(self, prev_sym, next_sym):
        self.prevSym = prev_sym
        self.nextSym = next_sym
        self.repeatCount = 0

    def __eq__(self, o):
        return isinstance(o, LogEvent) and self.prevSym == o.prevSym and self.nextSym == o.nextSym


class LogRingBuffer:
    def __init__(self, max_size):
        self.LOG_MAX_SIZE = max_size
        self.logBuffer = [None] * (max_size + 1)
        self.head = 1
        self.tail = 1
        self.logSize = 0
        self.overflowCount = 0

    def push(self, log_entry):
        self.logBuffer[self.tail] = log_entry
        if self.logSize >= self.LOG_MAX_SIZE:
            self.head = (self.head % self.LOG_MAX_SIZE) + 1
            self.overflowCount += 1
        else:
            self.logSize += 1
        self.tail = (self.tail % self.LOG_MAX_SIZE) + 1

    def increment_last(self):
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
        out = []
        idx = self.head
        for _ in range(self.logSize):
            out.append(self.logBuffer[idx])
            idx = (idx % self.LOG_MAX_SIZE) + 1
        return out


# =============================================================================
#  3. ActuatorCore (Soma FSM)
# =============================================================================
class ActuatorCore:
    def __init__(self, spike_width=1, refractory=3):
        self.SPIKE_WIDTH = spike_width
        self.REFRACTORY = refractory
        
        # CP
        self.enableRequest = False
        self.disableRequest = False
        
        # DP
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
                self.spikeTimer -= 1
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
                self.refractoryTimer -= 1
            if self.refractoryTimer == 0:
                self.status = ActuatorStatus.Disabled


# =============================================================================
#  4. SignalMgmtMC (Dendrite / Integrazione Continua)
# =============================================================================
class SignalMgmtMC:
    def __init__(self, n_masters, threshold=1000.0, ceiling=1700.0, floor=-700.0, leak_rate=120.0):
        self.LEAK_RATE = leak_rate
        self.THRESHOLD = threshold
        self.CEILING = ceiling
        self.FLOOR = floor
        self.WINDOW = 4

        self.accumulator = 0.0
        self.enableRequest = False
        self.enabledSensor = False
        
        # DP/CP
        self.n_masters = n_masters
        self.probe_ring = [0] * self.WINDOW
        self.probed = 0
        
        # Omeostasi
        self.actual_rate = 0.0
        self.SET_RATE = 0.10
        self.RATE_TAU = 20.0
        self.HUNGRY_DROP = 350.0
        self.hungry = False
        
        # Albero Dendritico (Rami)
        self.n_branches = 4
        self._branch_exc = [0.0] * self.n_branches
        self.LOCAL_TH = 1200.0     # Soglia NMDA locale del ramo
        self.LOCAL_BOOST = 1500.0  # Amplificazione se supera soglia locale
        
        self._arr = 0
        self._i_inh = 0.0
        self._i_shunt = 0.0

    def eff_threshold(self):
        return self.THRESHOLD - (self.HUNGRY_DROP if self.hungry else 0.0)

    def deposit(self, master_bit, charge, tag, mod_gain=0.25, branch_id=0):
        g = 1.0 if (tag & DRIVER) else mod_gain
        base_tag = tag & ~DRIVER
        if base_tag & E_SITE:
            self._branch_exc[branch_id] += charge * g
        elif base_tag == TX_SHUNT:
            self._i_shunt += charge * g
        else:
            self._i_inh += charge * g
        self._arr |= (1 << master_bit)

    def run(self):
        # bAP: il neurone ha sparato
        if self.enabledSensor:
            self.accumulator = 0.0
            
        # Servizio della coda (leakage verso zero)
        if self.accumulator > 0.0:
            self.accumulator = max(0.0, self.accumulator - self.LEAK_RATE)
        elif self.accumulator < 0.0:
            self.accumulator = min(0.0, self.accumulator + self.LEAK_RATE)
            
        # Albero Dendritico: Somma non-lineare dei rami eccitatori
        total_exc = 0.0
        for i in range(self.n_branches):
            b_exc = self._branch_exc[i]
            if b_exc > self.LOCAL_TH:
                total_exc += b_exc + self.LOCAL_BOOST  # NMDA spike
            else:
                total_exc += b_exc
            self._branch_exc[i] = 0.0  # Reset del ramo
            
        # Depositi e prelievi
        self.accumulator = min(self.CEILING, self.accumulator + total_exc)
        self.accumulator = max(self.FLOOR, self.accumulator - self._i_inh)
        
        # Shunting Inhibition (Normalizzazione Divisiva)
        if self._i_shunt > 0:
            self.accumulator /= (1.0 + self._i_shunt / 1000.0)
        
        # MC_TouchProbe (aggiorna maschere in finestra)
        self.probe_ring.pop(0)
        self.probe_ring.append(self._arr)
        self.probed = 0
        for m in self.probe_ring:
            self.probed |= m
            
        # Enable Request
        fired = self.accumulator >= self.eff_threshold()
        self.enableRequest = fired
        
        # Omeostasi (Hungry)
        self.actual_rate += ((1.0 if fired else 0.0) - self.actual_rate) / self.RATE_TAU
        self.hungry = (self.actual_rate < self.SET_RATE)
        
        self._arr = 0
        self._i_inh = 0.0
        self._i_shunt = 0.0


# =============================================================================
#  5. AxonDelay (Assone)
# =============================================================================
class AxonDelay:
    def __init__(self, dead_time_scan):
        self.buf = [0.0] * max(1, dead_time_scan)
        self.idx = 0

    def update(self, value_in):
        value_out = self.buf[self.idx]
        self.buf[self.idx] = value_in
        self.idx = (self.idx + 1) % len(self.buf)
        return value_out


# =============================================================================
#  6. Synapse (Camma Elettronica / Covarianza)
# =============================================================================
class Synapse:
    def __init__(self, transmitter=TX_EXC, weight=1.0):
        self.QUANTUM = 150.0
        self.TAU_RATE = 12.0
        self.TAU_MEAN = 80.0
        self.ETA_BASE = 25.0
        self.ETA_MOD = 6000.0
        self.ELIG_LEAK = 0.92
        self.W_MAX = 5.0
        self.W_MIN = 0.0
        self.transmitter = transmitter
        
        # Short-Term Plasticity (STP)
        self.U = 0.5           # Frazione di risorse usate allo spike
        self.TAU_REC = 15.0    # Tempo di recupero delle risorse (scan)
        self.R = 1.0           # Risorse disponibili (Coda vescicolare)
        
        self.weight = weight
        self.pre_rate = 0.0
        self.post_rate = 0.0
        self.pre_mean = 0.0
        self.post_mean = 0.0
        self.elig = 0.0

    def _leaky(self, avg, x, tau):
        return avg + (x - avg) / tau

    def update(self, pre_event, post_spike, modulator):
        pe = 1.0 if pre_event else 0.0
        po = 1.0 if post_spike else 0.0
        
        self.pre_rate = self._leaky(self.pre_rate, pe, self.TAU_RATE)
        self.post_rate = self._leaky(self.post_rate, po, self.TAU_RATE)
        self.pre_mean = self._leaky(self.pre_mean, self.pre_rate, self.TAU_MEAN)
        self.post_mean = self._leaky(self.post_mean, self.post_rate, self.TAU_MEAN)
        
        cov = (self.pre_rate - self.pre_mean) * (self.post_rate - self.post_mean)
        self.elig = self.elig * self.ELIG_LEAK + cov
        
        self.weight += self.ETA_BASE * cov
        if modulator != 0.0:
            self.weight += self.ETA_MOD * modulator * self.elig
            
        self.weight = max(self.W_MIN, min(self.W_MAX, self.weight))

    def charge(self, pre_event):
        # Recupero continuo delle risorse vescicolari
        self.R += (1.0 - self.R) / self.TAU_REC
        
        if pre_event:
            # Deplezione delle risorse e calcolo del rilascio
            rel = self.U * self.R
            self.R -= rel
            return self.weight * rel * self.QUANTUM
        return 0.0


# =============================================================================
#  7. Neuron (Contenitore Globale)
# =============================================================================
class Neuron:
    def __init__(self, name, n_masters, dead_time=1):
        self.name = name
        self.dendrite = SignalMgmtMC(n_masters=n_masters)
        self.soma = ActuatorCore()
        self.axon = AxonDelay(dead_time)
        self.axon_out = 0.0

    def run(self):
        # Esecuzione Dendrite -> Soma -> Axon
        self.dendrite.run()
        self.soma.enableRequest = self.dendrite.enableRequest
        self.soma.run()
        self.dendrite.enabledSensor = self.soma.enabledSensor
        self.axon_out = self.axon.update(1.0 if self.soma.enable else 0.0)
        return self.axon_out

    def reset_inputs(self):
        self.soma.disableRequest = False
        
