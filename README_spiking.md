# Istruzioni — `spiking_mnist_suite.py`

Test su **istanze spiking vere** (coda di Little scansione-per-scansione + FSM
ActuatorCore + refrattario), non il modello a frequenza. Due esperimenti:
MNIST e oblio catastrofico, con backprop come riferimento.

## 1. Requisiti

```bash
pip install numpy scikit-learn
```

## 2. Dati (MNIST) — una volta sola

Il codice cerca `mnist.npz` nella cartella; se non c'è, prova `openml`; se anche
quello è bloccato, usa il mirror GitHub di Nielsen. Il modo più robusto:

```bash
git clone --depth 1 https://github.com/mnielsen/neural-networks-and-deep-learning.git nnadl
```

Al primo avvio lo script legge `nnadl/data/mnist.pkl.gz` e salva `mnist.npz`
(così le volte dopo parte subito). In alternativa, se hai già MNIST come array,
salva tu un file `mnist.npz` con le chiavi `Xtr, ytr, Xte, yte` (pixel 0–1 o 0–255,
lo script normalizza da solo).

## 3. Esecuzione

```bash
python3 spiking_mnist_suite.py
```

Tempo: la **codifica spiking** è la parte lenta (ogni immagine viene simulata per
`T_STEPS` scansioni). Con i default (4000 neuroni, 80 scansioni, ~12k immagini)
sono ~1–3 minuti per esperimento su una CPU normale. È tutto vettorizzato su
batch, quindi scala bene con più core / con `T_STEPS` più basso.

## 4. Parametri regolabili (in cima a `main()`)

- `N_HID` (4000): neuroni di espansione (le cellule di Kenyon). Più alti = meglio, più lento.
- `SPARSITY` (0.10): frazione attiva dopo la k-WTA (l'inibizione). Più basso = codici più disgiunti.
- `T_STEPS` (80): scansioni per immagine. Più alto = conteggio spike più fine (più fedele al rate), più lento.
- `GAIN` (1.0): forza del drive. Basso = regime graduato; alto = satura (refrattario).
- `N_TRAIN`/`N_TEST` (12000/2000): dimensione del sottoinsieme.

## 5. Cosa aspettarsi (i numeri che ho ottenuto io)

**Esp. 1 — MNIST, ponte spiking↔rate:**

| codice | readout prototipo | readout delta locale |
|---|---|---|
| SPIKING (istanze vere) | ~84% | ~94% |
| RATE (mean-field) | ~84% | ~95% |

→ **Il ponte tiene**: lo spiking dà quasi lo stesso numero del suo mean-field a
frequenza (differenza < 1 punto). Il substrato biologico calcola la stessa cosa.

**Esp. 2 — oblio catastrofico (impara 0–4, poi 5–9, rimisura 0–4):**

| metodo | A dopo A | A dopo B | oblio |
|---|---|---|---|
| NOSTRO — prototipo (locale puro) | ~93% | ~87% | **~7 punti** |
| NOSTRO — delta locale | ~96% | ~23% | ~74 punti |
| BACKPROP (vanilla) | ~99% | ~0% | **~99 punti** |

→ Il **readout prototipo** (Hebbiano puro: scrive solo le colonne della classe
corrente) dimentica ~7 punti contro i ~99 di backprop — **~15× meno oblio**,
senza nessuna macchina speciale (no EWC, no replay). Il delta locale sta in mezzo.

## 6. Come leggerli — onestà

- **Ponte confermato**: "biologicamente fedele" qui è *letterale*, non un modo di
  dire — le istanze spiking eguagliano il loro limite a frequenza.
- **Compromesso reale**: l'apprendimento locale (prototipo) compra l'immunità
  all'oblio al prezzo di ~10 punti di accuratezza di picco rispetto al delta/
  backprop. Non è "vinciamo sempre" — è "vinciamo dove backprop è debole
  (continuo, robusto, locale), perdiamo dove è forte (picco su compito statico)".
- Su MNIST *statico* backprop resta avanti sul picco. Sul *continuo* il nostro
  circuito regge dove backprop collassa. Sono due assi diversi, misurati onestamente.

## 7. Nota tecnica

La codifica spiking simula ogni immagine come corrente costante (dopo espansione
casuale fissa + k-WTA) per `T_STEPS` scansioni, contando gli spike. Il refrattario
limita il rate massimo, quindi conteggi molto alti saturano: per un rate graduato
tieni `GAIN` basso. Il readout non usa mai backprop: prototipo = media dei codici
per classe (locale, per-classe indipendente); delta locale = one-vs-rest con
errore per-uscita (locale). L'unico blocco che usa backprop è `MLPClassifier`,
che è lì apposta come riferimento da battere sull'oblio.
