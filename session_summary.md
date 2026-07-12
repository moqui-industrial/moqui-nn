# Session Summary - Risoluzione Apprendimento SNN & Modellazione a Eventi Discreti

Questo documento riassume i lavori eseguiti, le decisioni di design architettonico e i risultati ottenuti in questa sessione di pair programming sul progetto **Biological SNN in JAX**.

---

## 🚀 1. Risoluzione dei Bug di Apprendimento (MNIST Accuracy ~10% -> 47.50%)

Abbiamo analizzato il comportamento della rete vettorializzata JAX (`moqui_nn`) confrontandola con il modello industriale di riferimento (`neuron_industrial.py`) ed identificato i bug bloccanti dell'apprendimento:

1.  **Bug Cascata FSM (Neuron)**: In `actuator_step` ([moqui_nn/Neuron.py](file:///home/igor/development/projects/nn/moqui_nn/Neuron.py)), le transizioni di stato dell'attuatore avvenivano a cascata nello stesso ciclo di scansione. Abbiamo corretto la logica forzando l'uso dello stato consolidato al ciclo precedente (`status_init = state.status`).
2.  **Bug Canale Sinaptico (Synapse)**: Le costanti sinaptiche di trasmissione `PACKETS_PER_SPIKE` e `PROCESSING_RATE` erano cablate fisse a 150 e 300, bloccando la trasmissione per pesi superiori a 2.0. Le abbiamo rese dinamiche e proporzionali al peso reale (`weight * QUANTUM`).
3.  **Congelamento Hebbiano via Kanban**: Le modifiche ai pesi sinaptici erano bloccate da `kanban_requests` (omeostasi del dendrite). Sotto *Teacher Forcing*, costringendo il neurone a sparare, la richiesta omeostatica crollava a zero, bloccando l'apprendimento. Abbiamo rimosso questo gating sinaptico artificiale.
4.  **Cancellazione dei Pesi via BCM**: La sottrazione dell'oblio dinamico BCM azzerava sistematicamente i pesi a ogni epoca. Abbiamo ripristinato la regola di covarianza pura (`net_plasticity = phase_matrix`) allineata alla biologia ed al modello industriale.
5.  **Allineamento Costanti STDP**: Corretto le costanti temporali dei filtri sinaptici sui valori industriali reali (`rate_tau = 12.0`, `mean_tau = 80.0`).
6.  **Reset della Dinamica Temporale**: Aggiunto un reset mirato degli accumulatori temporali (`reset_neuron_dynamics`) tra la presentazione di ciascuna immagine nei test per prevenire il carry-over dello stato.

Con queste correzioni, l'accuratezza su MNIST Digits è salita immediatamente al **47.50%** (utilizzando una riscalatura temporanea `/ 6.0` dei pesi per evitare la saturazione in inferenza).

---

## 🧠 2. Il Passaggio ai Fenomeni Logici Bottom-Up (Event Bookkeeping)

Su richiesta di eliminare ogni formula matematica ad-hoc (`/ 6.0`) e di modellare i fenomeni in chiave di **teoria delle code ed eventi discreti (stile PLC/automazione)**, abbiamo stabilito un quadro filosofico:
*   *Non modelliamo la fisica microscopica dei materiali (PDE).*
*   *Modelliamo la logica degli eventi transazionali e i tempi morti (come nell'astrazione ActuatorCore del motore asincrono).*

Abbiamo implementato due nuovi fenomeni biologici strutturati come code e stati logici discreti:

### A. Refrattarietà Assonale (Packet Drop Queue)
*   Introdotto `axon_refractory_timer` [in cicli di scansione] per ciascun assone in [moqui_nn/DeviceManager.py](file:///home/igor/development/projects/nn/moqui_nn/DeviceManager.py).
*   Se un neurone spara ma il suo assone è refrattario (timer > 0), lo spike viene scartato (packet drop stocastico). Se si propaga, blocca il canale per `2` cicli.
*   *Risultato*: Riduce drasticamente la sincronizzazione epilettica e favorisce la competizione Winner-Take-All temporale.

### B. Normalizzazione Eterosinaptica (Synaptic Heijunka)
*   Implementato un regolatore di budget a somma costante in [moqui_nn/Synapse.py](file:///home/igor/development/projects/nn/moqui_nn/Synapse.py).
*   A ciascun ciclo, se la somma dei pesi plastici in ingresso a un neurone supera il budget massimo (`max_total_capacity = 55.0`), questi vengono riscaldati proporzionalmente.
*   *Risultato*: Mantiene l'energia della rete bilanciata ed elimina qualsiasi divisione artificiale a posteriori dei pesi. Il valore di `55.0` emerge matematicamente dal bilanciamento della soglia ($1000.0$) rispetto alla sparsità media dell'input ($25\%$).

### C. Coda di Ritardo della Dopamina
*   La dopamina è stata trasformata da variabile istantanea globale in una **linea di ritardo di trasporto (buffer FIFO a 5 cicli)** in `DeviceManager.py`, simulando il tempo di viaggio chimico del segnale di errore.

---

## 🏆 3. Risultati Finali Ottenuti (MNIST Accuracy -> 54.50%)

Dopo l'implementazione dei meccanismi bottom-up e l'eliminazione dei termini industriali impropri (*Kanban/Heijunka*) dal codice, abbiamo rieseguito i test:

*   `test_learning.py` (Pavlovian Conditioning): **PASS** (peso appreso offline a `10.91`, risposta di fuga attiva).
*   `test_mnist_snn.py` (Classificazione pura): **PASS**. **Accuratezza record del 54.50%**, ottenuta in modo autonomo, auto-regolato e biologicamente coerente.

*Il quadro teorico completo e le analogie ingegneristiche (es. ActuatorCore) sono stati documentati nel file [philosophical_framework.md](file:///home/igor/.gemini/antigravity-ide/brain/18f5f2a8-3926-4606-8d5a-fb595ee55e11/philosophical_framework.md) negli artifacts.*
