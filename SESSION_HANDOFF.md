# Session Handoff — Reservoir Readout Debugging (2026-07-13)

Questo documento serve a trasferire il contesto completo di questa sessione su un'altra macchina. Non ripete quello che è già in `session_summary.md` (sessione precedente, architettura base) — copre tutto quello che è successo **dopo**, in una sessione molto più lunga e più profonda.

## 0. Come usare questo documento

Se stai leggendo questo su una macchina nuova (Claude Code o umano): il codice è già tutto qui, commitato e pushato su `github.com/moqui-industrial/moqui-nn` (branch `main`). Questo file spiega **perché** il codice è fatto così, cosa è stato provato e **scartato** (per non riprovarlo), e qual è il problema aperto.

## 1. La filosofia metodologica stabilita in questa sessione (fondamentale)

L'utente ha imposto un vincolo metodologico molto preciso, verificato e corretto più volte durante la sessione:

- **Niente formule analogiche/continue.** Solo teoria delle code, buffer, finestre temporali, confronti contro soglie (esattamente come `tempOverMax := T >= TMax` nel PLC originale dell'utente). Ogni volta che ho introdotto un decadimento esponenziale continuo (`peso * (1 - decay)`) l'utente lo ha bocciato e mi ha chiesto di riscriverlo come contatore intero confrontato con una soglia + pacchetto fisso scaricato. Vedi `staleness` in `Synapse.py` come esempio corretto.
- **Ogni meccanismo deve avere capacità E tempo di servizio finiti** (una vera coda). Esempi nel codice: `spike_timer`, `refractory_timer`, `axon_refractory_timer`, `dominance_timer`, `fatigue` (con tetto), `q_current`/`q_target`, `staleness`.
- **Il progetto è un porting fedele di pattern PLC/IEC 61131-3 reali**, presi da `iec61131/moqui/` (framework HVAC/robotica in Structured Text, escluso dal repo via `.gitignore` — è materiale di riferimento dell'utente, non nostro). Pattern chiave usati come modello:
  - `Actuator.st`: FSM Disabled→EnablePhase→Enabled→DisablePhase con vero handshake `enableRequest`/`disableRequest` (non un timeout incondizionato).
  - `SignalMgmt.st`: latch che si accende su una condizione e resta acceso finché non arriva un reset esplicito.
  - `LevelCtrlStatusMgr.st` / `EdgeTrigStatusMgr.st`: pattern livello-controllato vs fronte-triggerato.
  - `MoquiStart.st` → `InputSignalUpdate` → `Main` → `OutputSignalUpdate`: **ingressi congelati per l'intero scan, commit unico a fine ciclo** (semantica Jacobi, non Gauss-Seidel). Questo principio ha fatto **annullare** un'intera riscrittura (vedi sezione 3.4).
- **Verificare sempre empiricamente**, non fidarsi della teoria: ogni fix è stato validato rilanciando i test, e più volte i numeri hanno smentito le mie ipotesi (vedi sezione 4).

## 2. Stato del repository

- Remote: `git@github.com:moqui-industrial/moqui-nn.git`, branch `main`.
- Identità git **locale al repo** (non globale): `moqui-industrial <moqui-industrial@users.noreply.github.com>`. La chiave SSH nella macchina originale è già associata a questo account GitHub.
- `.gitignore` esclude `iec61131/` (materiale di riferimento PLC dell'utente) e `nnadl/` (clone di terze parti, rigenerabile via il comando in `README_spiking.md`).
- Test consolidati in `test/` (spostati in questa sessione): `test_learning.py`, `test_mnist_snn.py`, `test_mnist_reservoir.py`, `test_mnist_ann.py`, `test_jax_engine.py`, `industrial_snn_tests.py`. **Vanno lanciati dalla radice del progetto**, es. `python3 test/test_mnist_snn.py` (non da dentro `test/`).
- Rimossi in questa sessione (confermato: nessun riferimento residuo): `old/` (17 prototipi superati), `neuron.py`, `neuron_vectorized.py`, `hermaphrodite_chemical.csv.zip` (duplicato zippato di `datasets/*.csv`).

## 3. Cosa è stato costruito/corretto in questa sessione, in ordine cronologico

### 3.1 Fix al motore condiviso (`moqui_nn/`), tutti validati senza regressioni
1. **Fatica con capacità finita** (`Neuron.py`, `dendrite_step`): prima poteva crescere senza limite con spike ravvicinati; ora `fatigue_capacity` (default 1500.0) la satura, come un vero meccanismo di adattamento (canali K+ Ca-dipendenti).
2. **Normalizzazione eterosinaptica a scala temporale configurabile** (`Synapse.py`/`NNFacade.py`): nuovo `homeostatic_tau` (default 1.0 = comportamento originale istantaneo). Permette di rappresentare lo synaptic scaling come processo lento, se serve.
3. **Actuator con handshake vero** (`Neuron.py`, `actuator_step`): `Enabled` non scade più con un salto di stato incondizionato — resta finché non arriva un `disableRequest`, esterno o auto-emesso dal soma stesso allo scadere del proprio tempo di depolarizzazione, passando dallo stesso canale invece di scavalcarlo. Comportamento identico per `spike_width=1` (l'unico valore usato ovunque).

### 3.2 Reservoir sperimentale (`test/test_mnist_reservoir.py`): ricablatura biologicamente fedele
- **Convergenza sensori→hidden**: da densità 30% casuale a **K=6 connessioni fisse per neurone** (dato anatomico reale: cellule di Kenyon nella *Drosophila*, Caron et al. 2013), con guadagno per sinapsi ridotto (0-12 invece di 0-35) così che serva la coincidenza di più input, non un singolo canale forte.
- **Impulso sensoriale transitorio** invece di livello sostenuto: il pixel resta acceso solo nei primi `PULSE_DURATION=5` cicli su 20-40, come un fotorecettore reale che si adatta e smette di rispondere a luce costante (invece di dare tempo indefinito a qualunque debole eccitazione di superare soglia).
- Risultato di questi due fix: il codice hidden è passato da ~90% dei 500 neuroni attivi per immagine a **1.1-1.2% di duty cycle** (sparsità realistica) con una prima, reale differenza tra classi (Jaccard 0.315 stessa classe vs 0.273 classi diverse — margine sottile, ~15% relativo).

### 3.3 Gettone di dominanza (coda a capacità/tempo finiti in `DeviceManager.py`)
Per risolvere il "carosello a rotazione" (l'inibizione laterale basata sullo spike grezzo dura 1 solo ciclo, quindi il vincitore va in refrattario e libera il campo al prossimo in classifica): un gettone `dominance_timer` che si ricarica a un valore proporzionale al margine di vittoria (in "pacchetti" di sovrasoglia, tetto configurabile per popolazione via `topology.dominance_hold`) e scarica di 1 a ciclo. **Bug trovato e corretto**: la ricarica poteva arrotondare a zero per vincitori marginali, rendendo il tetto configurato inutile — corretto garantendo almeno 1 ciclo di mantenimento per qualunque vittoria (`token_gain = min(dominance_hold, 1 + surplus_packets)`).

### 3.4 Tentativo di risoluzione sequenziale intra-ciclo — IMPLEMENTATO POI ANNULLATO
Diagnosticato che con pesi solo moderatamente differenziati, **tutti i 10 motori possono superare soglia esattamente nello stesso ciclo** (misurato direttamente: tutti sparano al ciclo 10, identico), perché l'inibizione ha sempre 1 ciclo di ritardo di trasmissione — nessuno può inibire l'altro nell'istante esatto in cui pareggiano. Ho implementato una risoluzione sequenziale (Gauss-Seidel, `jax.lax.fori_loop` per-neurone dentro `network_step`) che risolveva il pareggio — **ma l'utente ha correttamente fatto notare che questo tradisce la semantica reale del PLC** (`InputSignalUpdate`→`Main`→`OutputSignalUpdate`: ingressi/stati congelati per l'intero scan, commit unico alla fine — esattamente Jacobi, non Gauss-Seidel). Un pareggio simultaneo, in quel modello, è un **sintomo onesto** di pesi non ancora differenziati, non un bug di esecuzione da mascherare con un ordine arbitrario. **La riscrittura è stata interamente annullata**, tornando al `network_step` vettorializzato originale. `neuron_step` è tornato a restituire solo `NeuronState` (non più una tupla con `enable_request`). **Non riprovare questa strada** — è stata esplicitamente giudicata scorretta dal punto di vista architetturale, non solo lenta (~10x).

### 3.5 Regola Delta Locale (readout alternativo alla covarianza pura)
Nuova modalità opt-in in `Synapse.py` (`use_delta_rule=True` in topologia), pensata come azione integrale su code (non un gradiente/backprop):
```
q_corrente[motore]: coda a tetto finito, +1 se il motore ha sparato questo ciclo, -1 altrimenti
q_target[motore]:   stessa natura, +1 se il bersaglio esterno chiede "dovresti sparare ora"
errore = q_target - q_corrente   (sottrazione intera di due code)
peso += Ki*Ts*errore, gated da idoneità presinaptica e da dopamina>0 (solo in training)
```
Corrisponde all'azione integrale di un PID (setpoint=target, PV=q_corrente, variabile manipolata=il peso). Il `target_signal` è un nuovo input per-ciclo di `network_step`/`run_network_scan` (default `None`→zeri, nessun impatto sui test esistenti).

### 3.6 Attenzione top-down — testata, nessun effetto
Un motore dominante inietta corrente eccitatoria all'indietro sui neuroni hidden da cui dipende di più (riusa la stessa matrice dei pesi imparati, letta al contrario — `attention_gain` in topologia, attivo solo quando dopamina≤0, cioè fuori dal training). **Risultato: zero effetto misurabile**, perché il pareggio tra motori era così simultaneo/esatto (vedi 3.4) che non c'era nessun vantaggio iniziale da amplificare — un meccanismo reattivo non può creare un margine che non esiste. Il meccanismo resta nel codice (default `attention_gain=0.0`, no-op), ma la sua inefficacia ha confermato che il problema è sempre stato a monte (readout), mai nella competizione a valle.

### 3.7 Diagnosi dei 3 problemi strutturali della regola delta (fondamentale per capire il problema aperto)
1. **Windup dell'integratore puro**: una volta che `q_corrente` insegue `q_target` (il motore bersaglio spara in modo affidabile grazie al teacher forcing), l'errore va a zero e la manopola smette di correggere — non scende mai più. Comportamento di rinforzo a senso unico, non un vero controllore bidirezionale.
2. **Nessuna visibilità incrociata tra selettori**: durante il training, i motori non-bersaglio erano soppressi esternamente così duramente (`ext_inh=800`) da non sparare mai — il loro errore restava a zero, nessuna correzione (positiva o negativa) sulle loro sinapsi. Fix applicato: **rimossa la soppressione esterna sui motori non-bersaglio** durante il training (ora è tutta a zero) — se un motore sbagliato spara comunque (ambiguità reale), quello è un segnale di errore negativo genuino (depressione eterosinaptica), non silenzio.
3. **Saturazione al tetto senza oblio**: i pesi che raggiungono `w_max=20` ci restano per sempre, appiattendo la differenza tra "fortemente diagnostico" e "occasionalmente utile". Fix: **stagionatura per-sinapsi** (`staleness` in `SynapseState`, campo `[N,N]` int32): conta i cicli dall'ultimo evento di rinforzo/punizione su QUELLA connessione; quando supera una soglia intera (`decay_period`, confronto esplicito, non formula), scarica un pacchetto fisso (`decay_quantum`). **Attenzione**: la prima versione di questo fix usava un decadimento moltiplicativo continuo (`peso *= (1-0.01)`) — l'utente l'ha bocciata esplicitamente come "formula analogica" e ho dovuto riscriverla nella forma a soglia intera. La versione a soglia intera è quella nel codice ora.

### 3.8 Budget presinaptico (normalizzazione eterosinaptica complementare)
Il budget esistente limitava solo QUANTO ARRIVA a un motore (per colonna). Aggiunto `max_presynaptic_capacity` in `Synapse.py`/`NNFacade.py`: limita anche QUANTO ESCE da un singolo sensore verso tutti i motori insieme (somma per riga, non per colonna) — costringe un sensore ambiguo a specializzarsi invece di accumulare un po' di peso verso 6-7 motori diversi. Default `1e6` (no-op).

### 3.9 Deviazioni teoriche esplorate e scartate (Information Geometry, funzioni ipergeometriche)
L'utente ha proposto due framework matematici avanzati (Geometria Informativa di Amari/metrica di Fisher/natural gradient; ricorsione ipergeometrica come "servo drive"/camma elettronica programmabile) chiedendo se potessero sbloccare l'accuratezza. Entrambi analizzati onestamente e **scartati come non applicabili** a questo problema specifico:
- **Information Geometry / natural gradient**: richiede ottimizzazione differenziabile a gradiente — il nostro readout è a code discrete, non backprop. Anche prescindendo da questo, il natural gradient migliora *come* ci si muove nello spazio dei pesi, non crea informazione che non c'è: se il codice hidden non separa le classi, la metrica di Fisher lungo quelle direzioni è essa stessa degenere.
- **Funzioni ipergeometriche / cluster-come-servo-drive**: tecnicamente costruibile con i nostri primitivi a coda (un guadagno che dipende da un contatore di passo, esattamente come la ricorsione ipergeometrica), ma agirebbe sulla **forma della risposta nel tempo** di un canale, non su **quali pixel un neurone ascolta** — e le nostre immagini sono statiche (impulso costante per 5 cicli), senza struttura temporale da sfruttare. L'unico pezzo onestamente riusabile: la **distribuzione ipergeometrica** (estrazione senza reimmissione) è il modello statistico esatto per la nostra convergenza K=6 (6 sensori scelti su 64 senza reinserimento) — utile come strumento di calibrazione analitica di K/soglia in futuro, non come fix.

### 3.10 Inibizione centro-periferia tra sensori (decorrelazione a monte) — testata, nessun effetto sull'accuratezza
Diagnosticato che il vero divario con una rete biologica non è un meccanismo mancante nel readout, ma uno **stadio mancante a monte**: il corpo fungino reale non cablega il livello di espansione casuale (K=6) a sensori chimici grezzi, ma a **glomeruli già preelaborati/decorrelati** (l'antennal lobe). Per la visione, l'equivalente è la retina (centro-periferia antagonista). Il nostro reservoir salta questo stadio: K=6 pesca direttamente da 64 pixel grezzi, correlati tra loro (regioni vicine di un tratto).

Implementato: inibizione laterale fissa tra sensori vicini sulla griglia 8×8 (stesso canale `w_inh` già usato altrove, mai cablato prima a livello sensori). **Trovato un problema di finestra temporale**: con `PULSE_DURATION=5` e guadagno pixel originale (300), un pixel al massimo supera soglia solo all'ultimissimo ciclo utile — zero margine perché la competizione centro-periferia agisca (confermato: guadagno di inibizione ×10 non cambiava nulla, byte-per-byte identico). Alzato `PIXEL_GAIN` a 600 per lasciare margine temporale reale. Su un piccolo campione diagnostico (8 immagini) il codice hidden cambiava misurabilmente (Jaccard 0.593/0.506 invece di 0.315/0.273, margine relativo simile ~15-17%). **Ma sul test completo (200 immagini): accuratezza invariata al 9.00%, pattern motorio identico byte-per-byte al run precedente senza centro-periferia.** Duty cycle salito a 2.2% (da 1.2%) per via del guadagno pixel più alto.

**Diagnosi**: le cifre scritte a mano condividono struttura anche a livello di bordi (non solo di regioni uniformi) tra classi diverse — a differenza degli odori, chimicamente quasi ortogonali, i bordi di "3" e "8" non lo sono. La decorrelazione locale rimuove ridondanza *dentro* un'immagine, non la sovrapposizione strutturale *tra* classi. Il pattern motorio identico al run precedente suggerisce inoltre che la dinamica del readout (stagionatura/budget) stia determinando l'esito più del codice hidden a monte, a questo punto della sessione.

## 4. Risultati empirici del reservoir — tabella completa (importante: non ripetere questi esperimenti, i risultati sono già noti)

Tutti con N_HIDDEN=500 salvo dove indicato, 1000 immagini training × 2 epoche, 200 immagini test (sklearn digits, non MNIST vero):

| Configurazione | Pesi motore (immagine 0) | Accuratezza | Note |
|---|---|---|---|
| Reservoir originale (denso 30%, pesi 0-35, livello sostenuto) | tutti sparano, non uniforme ma casuale | 6.00% | baseline iniziale, prima di questa sessione |
| + K=6, impulso transitorio, pesi ridotti | `[1,1,1,1,1,1,1,1,1,1]` | 9.00% | **scarica simultanea uniforme**, causa architetturale identificata (3.4) |
| + risoluzione sequenziale (poi annullata) | `[1,0,1,1,1,1,1,1,1,0]` | 9.00% | funzionava ma **architetturalmente scorretta**, annullata |
| + regola delta (decay=0, no presyn budget) | uniforme | 9.00% | nessun cambiamento da solo |
| + decadimento continuo 1% (bocciato, sostituito) | `8/10 vincono` poi `1/10` | 9% poi 1.5% | formula analogica, non nel codice finale |
| decay_period=300 (soglia intera) | `[1,1,1,1,1,1,1,1,1,1]` | 9.00% | troppo permissivo, nessun effetto |
| decay_period=100 | 9/10 pareggiati | 9.00% | ancora troppo permissivo |
| decay_period=30 | 5/10 pareggiati | 9.00% | primo miglioramento reale |
| decay_period=10 | 3/10 pareggiati, MA **uscita identica per 30 immagini diverse** | 9.00% | bug scoperto: la stagionatura aggressiva premia i sensori GENERALISTI (si attivano spesso) e punisce quelli specifici (si attivano di rado) — l'opposto di quanto voluto |
| decay_period=2 (estremo) | `[0,0,1,0,0,0,0,0,0,0]` netto ma sbagliato | 1.50% | erode anche il segnale corretto |
| **N_HIDDEN=2000** (cellule di Kenyon reali) con decay_period=10 | `[0,0,1,0,0,0,0,1,0,0]` | 10.00% | **falsifica l'ipotesi "serve più capacità"**: 4x neuroni, nessun miglioramento. Costo: 42 minuti di calcolo |
| Tornati a 500, decay_period=30 + presyn_budget=25 | `[1,0,1,1,0,0,1,1,0,0]` | 9.00% | budget troppo permissivo (verificato: p90 somma uscita = 10.3 dopo 300 immagini, il tetto 25 non vincolava quasi nessuno) |
| decay_period=30 + presyn_budget=12 | `[1,0,1,0,0,0,1,1,0,0]` | 9.00% | miglioramento marginale (4/10 pareggiati invece di 5), **accuratezza sempre ferma** |
| + centro-periferia sensori (SURROUND_GAIN=2000, PIXEL_GAIN=600) | `[1,0,1,0,0,0,1,1,0,0]` | 9.00% | **identico byte-per-byte al run precedente** nonostante il codice hidden misurabilmente diverso (duty cycle 2.2% invece di 1.2%) — la leva testata (decorrelazione a monte) non è servita, e l'identità del pattern suggerisce che il readout stesso domini l'esito |

**Conclusione onesta raggiunta a fine sessione (aggiornata dopo 3.9/3.10)**: nessuna combinazione di meccanismi lato readout (competizione, persistenza, attenzione, regola delta, stagionatura, budget presinaptico), l'aumento di capacità (4x neuroni), né la decorrelazione a monte (centro-periferia) ha spostato l'accuratezza dal livello del caso (~9-10%). Il sospetto più solido ora: le cifre scritte a mano condividono struttura (bordi, curve) tra classi diverse in un modo che né un'espansione casuale K=6 né una singola fase di decorrelazione locale possono sciogliere — a differenza degli odori (quasi ortogonali chimicamente), per cui questa intera ricetta (random+sparse+locale) è nata e funziona bene in natura. **Non ancora tentato**: un cablaggio strutturato/appreso (non casuale) per l'espansione stessa, o una decorrelazione più aggressiva/multi-scala (non solo centro-periferia a un passo).

## 5. Test di controllo, sempre validi (nessuna regressione durante tutta la sessione)
- `test/test_learning.py` (condizionamento pavloviano): peso appreso **10.91** dopo il replay in sonno, motore di fuga scatta. **Invariato dall'inizio alla fine della sessione.**
- `test/test_mnist_snn.py` (architettura diretta sensori→motori, senza reservoir/hidden layer): **54.50%** di accuratezza. Anche questo invariato — dimostra che il motore condiviso funziona correttamente, il problema è specifico dell'architettura a reservoir con hidden layer casuale.

## 6. Parametri correnti in `test/test_mnist_reservoir.py` (stato al termine della sessione)
```
N_HIDDEN = 500
K_CONVERGENCE = 6
HIDDEN_LATERAL_INHIB = 50.0
SYNAPTIC_BUDGET = 600.0
PULSE_DURATION = 5
PIXEL_GAIN = 600.0          # alzato da 300 per lasciare margine temporale alla competizione centro-periferia
SURROUND_GAIN = 2000.0      # inibizione laterale sensori vicini, griglia 8x8 (sezione 3.10)
HIDDEN_DOMINANCE_HOLD = 8
MOTOR_DOMINANCE_HOLD = 30
DELTA_RULE_DECAY_PERIOD = 30
DELTA_RULE_DECAY_QUANTUM = 1.0
MAX_PRESYNAPTIC_CAPACITY = 12.0
ATTENTION_GAIN = 0.5
```

## 7. Note su come l'utente preferisce lavorare (per continuità)
- Preferisce spiegazioni in termini di impianto industriale/PLC, non gergo neuroscientifico — è un esperto di automazione, non di neuroscienze.
- Corregge attivamente e con precisione qualunque scorciatoia numerica ("formula analogica") — vuole solo code, contatori, confronti a soglia.
- Vuole verifica empirica reale (rilanciare i test) prima di accettare una spiegazione teorica.
- Apprezza quando gli errori vengono ammessi esplicitamente e spiegati (es. il decadimento continuo, la riscrittura sequenziale poi annullata) piuttosto che minimizzati.
- I test lunghi (reservoir con 500 neuroni: ~90-120s; con 2000: ~40 minuti) vanno lanciati in background con notifica, non in blocco.
