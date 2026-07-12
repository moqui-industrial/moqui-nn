# Istruzioni — `neuron_full_suite.py` (COLLAUDO COMPLETO)

Questo è il collaudo della **macchina completa**: il `Neuron` intero con **tutti**
i sottosistemi attivi, istanziato N volte nel `SpikingNet` (il facade), su **tre**
compiti diagnostici scelti per far emergere i tre esiti — anche quello negativo.

## Sottosistemi attivi in ogni passo (niente spento)

coda di Little (fuga proporzionale/RC) · FSM ActuatorCore (spike one-shot +
refrattario) · E/I con neuroni inibitori reali (bilancio E-I → sparsità, **non**
`np.partition`) · sinapsi con peso e segno · **dead time assonico** (ring buffer
per-ritardo) · apprendimento a **tre fattori** (R-STDP: `dw = kPlast · modulatore
· elig_pre · spike_post`, **niente backprop**) · omeostasi/**HUNGRY** (soglia
adattiva) · **neuromodulazione** (kGain/kPlast/kExplore dal blackboard) ·
eligibilità.

## Requisiti e dati

```bash
pip install numpy scikit-learn
git clone --depth 1 https://github.com/mnielsen/neural-networks-and-deep-learning.git nnadl
```
(al primo avvio legge `nnadl/data/mnist.pkl.gz` e salva `mnist.npz`).

## Esecuzione

```bash
python3 neuron_full_suite.py
```

Tempo: la codifica spiking simula **ogni immagine** per `T_STEPS` scansioni con
tutti i meccanismi → è la parte lenta (~1-3 min totali coi default). Tutto è
regolabile in cima a ciascuna `task_*`: `n_hid`, `n_train`, `T_STEPS`.

## I tre compiti e gli esiti attesi (miei numeri, taglia ridotta)

| compito | nostro (macchina completa) | riferimento | esito |
|---|---|---|---|
| **1. MNIST statico** | ~84% (spiking + 3-fattori) | backprop ~92% | **PERDE sul picco** ✓ atteso |
| **2. Oblio catastrofico** | ~7 punti persi | backprop ~99 persi | **VINCE** |
| **3. Ordine temporale** | ~100% (dead time+coincidenza) | ANN feedforward ~51% (caso) | **VINCE per costruzione** |

## Perché proprio questi tre (dalla letteratura, non a caso)

- **MNIST**: feedforward, atemporale, supervisionato = terreno di casa di
  backprop. La macchina completa gira "coi freni tirati" (tempo, dead time,
  omeostasi non aiutano un compito senza tempo) e **deve** perdere sul picco.
  È il controllo che dimostra che non barіamo.
- **Oblio**: apprendimento sequenziale. Le ANN backprop soffrono oblio
  catastrofico (McCloskey & Cohen 1989). L'apprendimento locale + codici sparsi
  regge — vantaggio **strutturale** documentato.
- **Ordine temporale**: distinguere "A prima di B" da "B prima di A". Una ANN
  feedforward **non ha accesso al tempo**; le serve un'architettura ricorrente
  apposta. Lo spiking ha il tempo **nativo** (dead time + coincidenza). Il
  confronto è a nostro favore per costruzione — ed è il punto.

## Come leggerlo — è un COLLAUDO, non una vetrina

Tre esiti diversi da **una sola macchina completa**: perde (1), vince (2), vince
per costruzione (3). L'esito negativo di (1) **è emerso**, come deve in un
collaudo serio. Il valore del substrato biologico non è "impara meglio in
generale" — è "risolve nativamente ciò che backprop feedforward non può (tempo,
continuo, robustezza), al prezzo di un picco inferiore sul compito statico".

## Note tecniche

- Il readout di MNIST usa la regola a **tre fattori** (reward-modulata), non
  backprop; quello dell'oblio usa il **prototipo locale** (scrive solo le colonne
  della classe corrente → A non si dimentica); quello temporale è un percettrone
  locale sui rilevatori. L'unico blocco backprop è `MLPClassifier`, che è lì
  apposta come riferimento.
- Il dead time è realizzato con un **ring buffer** di spike (come il tuo
  `LogRingBuffer`): ogni sinapsi legge lo spike del pre col proprio ritardo.
- Tutto è vettorizzato in structure-of-arrays; per scalare, alza `n_hid` e
  `n_train`; per accelerare, abbassa `T_STEPS` (meno risoluzione di rate).
