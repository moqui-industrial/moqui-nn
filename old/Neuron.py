#!/usr/bin/env python3
# =============================================================================
#  Neuron.py  -  il neurone completo, con la teoria dentro
# =============================================================================
#
#  Questo file NON e' un classificatore ne' un benchmark. E' il neurone nella
#  sua forma piena, con TUTTI i meccanismi, e con la spiegazione scritta accanto
#  al codice - perche' il codice E' la tesi. Ogni sottosistema qui sotto e' un
#  pezzo del ragionamento che abbiamo costruito. Lo rimetto in ordine dall'inizio.
#
# -----------------------------------------------------------------------------
#  IL PRINCIPIO FONDANTE: LA CAMMA ELETTRONICA
# -----------------------------------------------------------------------------
#  Un neurone non e' un sommatore pesato. E' un ASSE che insegue un MASTER.
#  Nell'automazione, una camma elettronica lega un asse slave a un asse master:
#  lo slave segue la posizione del master secondo un profilo. Ma l'aggancio non
#  e' istantaneo - lo slave ha inerzia, e riesce a seguire il master solo se
#  entra "in fase" con lui entro una finestra di tolleranza. Se master e slave
#  sono in fase, l'aggancio tiene e si rinforza; se sono in antifase, l'aggancio
#  slitta e si allenta.
#
#  Il neurone e' questo. Il presinaptico e' il master, il postsinaptico e' lo
#  slave. Il peso sinaptico e' la FORZA DELL'AGGANCIO tra i due. E - punto
#  centrale, l'origine di tutto - quella forza NON e' data dall'esterno:
#
#        IL PESO NASCE DALLA SINCRONIA.
#
#  Il peso pre->post cresce quando lo spike del master e lo spike dello slave
#  CADONO INSIEME (in fase, dentro la finestra), e cala quando cadono separati
#  (in antifase). E' la regola di covarianza: dw ~ covarianza(pre, post). Nessun
#  peso e' assegnato; ogni peso e' il residuo storico di quante volte quei due
#  neuroni hanno "battuto insieme". La rete impara chi insegue chi.
#
# -----------------------------------------------------------------------------
#  LA FINESTRA TEMPORALE (l'inerzia = il cuscinetto d'aggancio)
# -----------------------------------------------------------------------------
#  Cosa vuol dire "insieme"? Lo definisce la FINESTRA TEMPORALE, che e' l'inerzia
#  della coda dendritica. Un evento in arrivo non viene letto e dimenticato:
#  entra nella coda e vi resta "vivo" per un tempo W (decadendo), la finestra
#  d'inerzia. Due eventi sono "in coincidenza" se cadono entrambi nella finestra
#  prima che il primo sia svanito.
#
#  Questa finestra e' il cuscinetto entro cui lo slave fa in tempo ad agganciare
#  il master. In teoria delle code e' il tempo di residenza W della legge di
#  Little (L = lambda * W): il potenziale L e' il tasso di arrivo lambda per
#  l'inerzia W. La coincidenza la fa il DECADIMENTO della coda, non un clock:
#  gli eventi si raggruppano perche' restano vivi insieme, non perche' un ciclo
#  li raccoglie. Ingresso a interrupt (asincrono), integrazione continua, uscita
#  a ritmo.
#
#  E il DEAD TIME (il ritardo dell'assone) deve stare DENTRO la finestra, o
#  l'aggancio non avviene: se il segnale del master arriva quando la finestra si
#  e' gia' svuotata, lo slave non lo vede mai in coincidenza. Vincolo fisico:
#        dead_time (ritardo assone)  <=  finestra d'inerzia (memoria della coda)
#
# -----------------------------------------------------------------------------
#  I DUE PIANI SOVRAPPOSTI (kanban: control plane su data plane)
# -----------------------------------------------------------------------------
#  Come nel kanban un flusso informativo (le carte) viaggia all'indietro sopra
#  il flusso di materiale, nel neurone il DATA PLANE porta gli spike in avanti e
#  il CONTROL PLANE porta le carte all'indietro. La fame omeostatica (HUNGRY) e'
#  la carta kanban: un neurone affamato "tira" segnale dai master abbassando la
#  soglia (pull, non push). La sinapsi E' il bin kanban per-master: il suo peso
#  e' la priorita' della carta, che si auto-ordina per sincronia.
#
# -----------------------------------------------------------------------------
#  IL MUSCOLO UNICO
# -----------------------------------------------------------------------------
#  Il neurone ha UNA azione: lo spike. Non spara "positivo" o "negativo": spara
#  o inibisce lo sparo. L'effetto opposto (il "caldo" nell'HVAC) lo ottiene NON
#  sparando. E/I non e' un peso negativo: e' a quale POOL appartiene il master
#  (eccitatorio o inibitorio), una proprieta' del presinaptico, non del legame.
#
# -----------------------------------------------------------------------------
#  I TRE FATTORI (come il peso si consolida)
# -----------------------------------------------------------------------------
#  La sincronia locale da sola dice "questi due battono insieme". Ma consolidare
#  o no quel legame lo decide un TERZO fattore globale: il modulatore (la deriva
#  dell'errore di sistema, la dopamina, "sto migliorando?"). La regola completa:
#
#        dw  =  modulatore  x  coincidenza_locale(pre,post)  x  esplorazione
#
#  "assonanza tra loop locale e loop generale": rinforza lo sparo solo quando la
#  coincidenza locale e' IN FASE con un miglioramento globale. L'eligibilita' e'
#  la traccia che tiene viva la coincidenza finche' il modulatore arriva (deve
#  coprire il dead time, o il credito si disallinea).
#
#  Sparare e imparare sono due tempi: prima agisci (spike per coincidenza), poi
#  - col senno del segnale globale - decidi se quel legame meritava rinforzo.
#
# -----------------------------------------------------------------------------
#  IL SUBSTRATO
# -----------------------------------------------------------------------------
#  Un solo primitivo: Actuator bistabile (FSM: Disabled/EnablePhase/Enabled/
#  DisablePhase -> spike one-shot + refrattario) + coda finita (bitfield). Il
#  continuo non e' primitivo: e' PRODOTTO dal discreto tramite la coda (legge di
#  Little). E' il quanto sinaptico di Katz: eventi vescicolari discreti sommati
#  in un potenziale graduato. Il neurone e' un DES che fabbrica il continuo.
# =============================================================================

import random


# --- costanti anatomiche (recipe di bootstrap: NON le tocca la neuromodulazione)
ENABLE_TIME = 2        # larghezza dello spike (scan)
DISABLE_TIME = 3       # refrattario (scan)
S_DISABLED, S_ENABLEPHASE, S_ENABLED, S_DISABLEPHASE = 0, 1, 2, 3


class Synapse:
    """L'aggancio a camma tra un master (pre) e questo slave (post).
    Il peso e' la FORZA DELL'AGGANCIO e NASCE dalla sincronia: cresce quando pre
    e post battono in fase, cala in antifase. E' anche il bin kanban per-master."""
    __slots__ = ("pre", "w", "sign", "delay", "elig_pre", "plastic")

    def __init__(self, pre, w0, excitatory=True, delay=1, plastic=True):
        self.pre = pre                 # indice del neurone master
        self.w = w0                    # forza dell'aggancio (nasce, non e' data)
        self.sign = 1.0 if excitatory else -1.0   # E/I = pool del master
        self.delay = delay             # dead time dell'assone (dentro la finestra!)
        self.elig_pre = 0.0            # traccia: coincidenza pre in attesa del modulatore
        self.plastic = plastic


class Neuron:
    """Il neurone completo: coda (finestra temporale) + Actuator (soma) + camme
    (sinapsi con peso da sincronia) + omeostasi + tre fattori + neuromodulazione."""
    __slots__ = ("L", "mu", "status", "timer", "threshold", "thr0",
                 "activity", "elig", "spike", "spikePrev", "hist", "hpos",
                 "is_inh", "syn", "TARGET_ACT", "HOMEO_RATE", "ELIG_DECAY", "COV_RATE")

    def __init__(self, tau_inertia=8.0, threshold=100.0, excitatory=True):
        # --- DENDRITE: la coda finita = la finestra temporale (inerzia W) ---
        self.L = 0.0                   # potenziale = riempimento coda (Little: L = lambda*W)
        self.mu = 1.0 / tau_inertia    # fuga proporzionale (1/W) = membrana RC
        # --- SOMA: Actuator bistabile ---
        self.status = S_DISABLED
        self.timer = 0
        self.threshold = threshold     # soglia adattiva (omeostasi/HUNGRY)
        self.thr0 = threshold
        self.activity = 0.0            # media di attivita' (per la fame)
        # --- traccia per i tre fattori ---
        self.elig = 0.0                # eligibilita' post (coincidenza in attesa)
        self.spike = False
        self.spikePrev = False         # l'assone consegna con >=1 scan di ritardo
        self.hist = [False] * 16       # storia spike (ring) per il DEAD TIME assonico
        self.hpos = 0
        self.is_inh = not excitatory   # a quale pool appartengo (E/I)
        # --- AXON/CAMME: le sinapsi in ingresso (gli agganci ai master) ---
        self.syn = []                  # lista di Synapse (i bin kanban per-master)
        # --- parametri di apprendimento ---
        self.TARGET_ACT = 0.05         # set-point omeostatico (fame)
        self.HOMEO_RATE = 0.02
        self.ELIG_DECAY = 0.80         # quanto vive la coincidenza (copre il dead time)
        self.COV_RATE = 0.02           # velocita' di nascita del peso dalla covarianza

    def delayed(self, d):
        # lo spike di 'd' scansioni fa (dead time dell'assone via ring buffer)
        return self.hist[(self.hpos - d) % len(self.hist)]

    def connect(self, pre, w0=0.0, excitatory=True, delay=1, plastic=True):
        # il peso PARTE quasi da zero e NASCE dalla sincronia (non e' assegnato)
        self.syn.append(Synapse(pre, w0, excitatory, delay, plastic))

    # ------------------------------------------------------------------ scan --
    def scan(self, population, ext_drive, kGain, kExplore, modulator, rng):
        """Una scansione. 'population' = lista di tutti i Neuron (per leggere gli
        spike dei master, ritardati = dead time). Tutti i meccanismi attivi."""

        # 1. DATA PLANE: i master consegnano lo spike RITARDATO (dead time).
        #    La carica entra nella coda e vi resta viva per la finestra d'inerzia.
        drive = ext_drive
        for s in self.syn:
            master = population[s.pre]
            # lo spike del master 'delay' scan fa (qui: spikePrev = 1 scan; per
            # ritardi maggiori la popolazione tiene un ring buffer - vedi facade)
            fired = master.spikePrev if s.delay <= 1 else master.delayed(s.delay)
            if fired:
                drive += s.sign * s.w      # E somma, I sottrae (muscolo unico)
                # la coincidenza pre e' arrivata: accendi la traccia della sinapsi
                s.elig_pre = 1.0
            else:
                s.elig_pre *= self.ELIG_DECAY   # svanisce dentro la finestra

        # 2. DENDRITE: coda di Little (integra in continuo, coincidenza per decadimento)
        self.L = self.L * (1.0 - self.mu) + drive
        if self.L < 0.0:
            self.L = 0.0
        self.spikePrev = self.spike
        self.spike = False

        # 3. NEUROMODULAZIONE: il gain abbassa la soglia efficace; l'esplora aggiunge rumore
        effThr = self.threshold / kGain if kGain > 0.0 else self.threshold
        driveEff = self.L + (rng.gauss(0.0, kExplore) if kExplore > 0.0 else 0.0)

        # 4. SOMA: FSM ActuatorCore (spike one-shot + refrattario)
        st = self.status
        if st == S_DISABLED:
            if driveEff >= effThr:
                self.status = S_ENABLEPHASE; self.timer = ENABLE_TIME
        elif st == S_ENABLEPHASE:
            if self.timer > 0: self.timer -= 1
            if self.timer == 0: self.status = S_ENABLED
        elif st == S_ENABLED:
            self.spike = True                      # SPIKE (muscolo unico)
            self.status = S_DISABLEPHASE; self.timer = DISABLE_TIME
        elif st == S_DISABLEPHASE:
            if self.timer > 0: self.timer -= 1
            if self.timer == 0: self.status = S_DISABLED

        # 5. OMEOSTASI / HUNGRY: la soglia insegue il set-point di attivita'.
        #    Coda troppo vuota -> affamato -> abbassa la soglia -> tira i master (pull).
        post = 1.0 if self.spike else 0.0
        self.activity += 0.02 * (post - self.activity)
        self.threshold += self.HOMEO_RATE * (self.activity - self.TARGET_ACT) * self.thr0
        if self.threshold < 5.0:
            self.threshold = 5.0

        # 6. ELIGIBILITA' post (traccia della mia coincidenza, in attesa del modulatore)
        self.elig = self.elig * self.ELIG_DECAY + post

        # 7. NASCITA DEL PESO DALLA SINCRONIA (la camma che si aggancia).
        #    Regola di COVARIANZA a tre fattori: il peso cresce se la coincidenza
        #    pre (s.elig_pre) supera una soglia THETA quando io sparo (potenziamento,
        #    in fase), e CALA se e' sotto soglia (depressione, antifase). E' la
        #    covarianza vera - non Hebb puro: dw ~ (coincidenza - THETA) x post,
        #    modulata dal terzo fattore. In fase sale, in antifase scende.
        if self.spike:
            m = modulator if modulator != 0.0 else 1.0   # senza reward: covarianza pura
            THETA = 0.5                                   # soglia BCM (media della coincidenza)
            for s in self.syn:
                if not s.plastic:
                    continue
                dw = self.COV_RATE * m * (s.elig_pre - THETA) * post
                s.w += dw
                if s.w < 0.0:
                    s.w = 0.0
            # normalizzazione omeostatica: tetto sulla somma dei pesi entranti
            tot = sum(s.w for s in self.syn if s.plastic)
            if tot > 1e-9:
                target = 4.0 * sum(1 for s in self.syn if s.plastic)
                f = target / tot
                if f < 1.0:
                    for s in self.syn:
                        if s.plastic:
                            s.w *= f

        # 8. scrivi lo spike nella storia (per il dead time del prossimo giro)
        self.hpos = (self.hpos + 1) % len(self.hist)
        self.hist[self.hpos] = self.spike
        return self.spike


# =============================================================================
#  DIMOSTRAZIONE: il peso NASCE dalla sincronia (in fase sale, in antifase no).
#  Due master (0 in fase con il post, 1 in antifase) agganciati allo stesso
#  slave (2). Nessun peso e' dato: partono da zero. Dopo l'esposizione, il peso
#  del master in fase e' cresciuto, quello in antifase no. E' la camma che si
#  aggancia solo a chi batte insieme.
# =============================================================================
def _demo_nascita_peso():
    rng = random.Random(0)
    pre_in_phase = Neuron(tau_inertia=8.0, threshold=1.0)     # master 0
    pre_anti = Neuron(tau_inertia=8.0, threshold=1.0)         # master 1
    post = Neuron(tau_inertia=8.0, threshold=1.0)             # slave 2
    post.connect(pre=0, w0=0.10, delay=1)                     # aggancio a 0 (nasce)
    post.connect(pre=1, w0=0.10, delay=1)                     # aggancio a 1 (nasce)
    post.COV_RATE = 0.05
    pop = [pre_in_phase, pre_anti, post]

    PERIOD = 10
    print("Nascita del peso dalla sincronia (entrambi partono da 0.10):\n")
    print("  scan |  peso<-master IN FASE  |  peso<-master ANTIFASE")
    for t in range(600):
        # master 0 e il post battono in fase; master 1 in antifase (sfasato di mezzo periodo)
        d0 = 100.0 if (t % PERIOD) == 0 else 0.0
        d1 = 100.0 if (t % PERIOD) == (PERIOD // 2) else 0.0
        dpost = 100.0 if (t % PERIOD) == 1 else 0.0     # il post segue il master 0 (in fase)
        pre_in_phase.scan(pop, d0, 1.0, 0.0, 0.0, rng)
        pre_anti.scan(pop, d1, 1.0, 0.0, 0.0, rng)
        post.scan(pop, dpost, 1.0, 0.0, 0.0, rng)       # modulator=0 -> Hebb puro
        if t % 120 == 0:
            print("  %4d |         %.3f          |        %.3f"
                  % (t, post.syn[0].w, post.syn[1].w))
    r = post.syn[0].w / max(post.syn[1].w, 1e-9)
    print("\n  -> il peso verso il master IN FASE cresce ~%.1fx quello ANTIFASE." % r)
    print("     Il peso NON e' dato: nasce dalla coincidenza pre-post nella")
    print("     finestra temporale ed e' GRADUATO dalla sincronia. E' la camma")
    print("     che si aggancia forte a chi batte insieme, debole a chi e' sfasato.")
    print("     Questo e' il principio fondante: la sincronia genera il legame.")


if __name__ == "__main__":
    _demo_nascita_peso()
