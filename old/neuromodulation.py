#!/usr/bin/env python3
# =============================================================================
#  neuromodulation.py   (FASE 4)
#  Neuromodulazione + instradamento, nell'architettura MULTI-TASK reale:
#  loop veloce (neuroni) + task LENTO che scrive il regime nel blackboard.
#  I due si parlano solo attraverso lo stato condiviso (come DeviceFacade),
#  disaccoppiati dai ritmi diversi - il blackboard e' il buffer che assorbe
#  la differenza di velocita' (camma/inerzia a livello di task).
#
#  NIENTE Main / MainRuleEngine: nessuna politica cablata. Il "modo" non e'
#  scritto da un progettista - e' solo il regime di neuromodulazione corrente,
#  un livello globale lento letto dai neuroni veloci (come il locus coeruleus).
#
#  Meccanismo (non politica): la neuromodulazione moltiplica parametri che
#  gia' esistono - non tocca i pesi appresi. Stessa rete, stessi pesi, macchine
#  diverse a seconda del regime.
#
#  Ritmi (come nel task config CODESYS): controllo veloce vs neuromod lenta.
#  Qui SLOW_PERIOD scan veloci per ogni aggiornamento del task lento.
# =============================================================================

import random

SLOW_PERIOD = 40      # scan veloci per ogni ciclo del task lento (10ms vs ~400ms)


# --- il blackboard condiviso (come DeviceFacade): solo stato, nessuna politica ---
class Blackboard:
    def __init__(self):
        # NeuromodConfig: i moltiplicatori globali (set atomico, scritto dal task lento)
        self.kGain = 1.0          # eccitabilita' (soglia)
        self.kPlasticity = 0.0    # tasso di apprendimento (impara / esegue)
        self.kExplore = 0.0       # rumore di esplorazione (esplora / sfrutta)
        self.regimeName = "esegue"
        self.gateSelect = 0       # instradamento: quale input passa
        # stato della rete (i pesi appresi vivono qui, la neuromod NON li tocca)
        self.w = 0.2              # peso appreso (mapping input->output)


# --- TASK LENTO: scrive il regime nel blackboard, al SUO ritmo ---------------
REGIMES = {
    #                 kGain  kPlast  kExplore
    "impara":        (1.0,   0.30,   0.15),   # setup: plastico, esplora
    "esegue":        (1.0,   0.00,   0.00),   # produzione: pesi fissi, deterministico
    "esplora":       (1.2,   0.30,   0.40),   # cerca varianti (alto rumore)
    "sfrutta":       (1.3,   0.05,   0.00),   # commette sul meglio, gain alto
}


def slow_task_neuromod(bb, regime):
    # scrive un SET COMPLETO ATOMICO (mai incrementale), come MqttParameterSub
    g, p, e = REGIMES[regime]
    bb.kGain = g; bb.kPlasticity = p; bb.kExplore = e
    bb.regimeName = regime


# --- TASK VELOCE: un neurone che legge il regime dal blackboard ogni scan -----
def fast_task_neuron(bb, x, target, rng, learn_base=0.5):
    # legge i parametri di neuromodulazione dal blackboard (non glieli passa nessuno)
    noise = rng.gauss(0.0, bb.kExplore)
    out = bb.kGain * bb.w * x + noise
    error = target - out
    # apprendimento modulato: LR effettivo = LR_base * kPlasticity (dal blackboard)
    lr = learn_base * bb.kPlasticity
    bb.w += lr * error * x                 # i pesi cambiano SOLO se plasticity > 0
    return out, error


def demo_neuromodulation():
    print("DEMO 1 - neuromodulazione: stessa rete, stessi pesi, regimi diversi")
    print("Il task lento cambia regime; il task veloce legge e cambia comportamento.\n")
    rng = random.Random(0)
    bb = Blackboard()
    # sequenza di regimi imposta dal task lento (uno ogni SLOW_PERIOD scan veloci)
    schedule = ["impara", "esegue", "esplora", "sfrutta", "esegue"]
    x, target = 1.0, 1.0                    # vogliamo che out seguano target=1.0
    log = []
    scan = 0
    for regime in schedule:
        slow_task_neuromod(bb, regime)      # TASK LENTO scrive il blackboard
        w_start = bb.w
        outs = []
        for _ in range(SLOW_PERIOD):        # TASK VELOCE gira molte volte per un ciclo lento
            out, err = fast_task_neuron(bb, x, target, rng)
            outs.append(out)
            log.append((scan, regime, bb.w, out, err))
            scan += 1
        import statistics as st
        dw = bb.w - w_start
        var = st.pvariance(outs)
        print("  regime '%-8s': peso %.2f->%.2f (delta %+.3f) | varianza uscita %.4f"
              % (regime, w_start, bb.w, dw, var))
    print("\n  -> 'impara' muove il peso; 'esegue' lo lascia fermo (delta 0) e l'uscita")
    print("     e' deterministica (varianza ~0); 'esplora' alza la varianza; 'sfrutta'")
    print("     commette senza rumore. Stessi pesi, comportamenti diversi: la manopola")
    print("     cambia COME lavora la rete senza riscrivere niente.\n")
    return log


# --- instradamento via inibizione-gate (attenzione) --------------------------
def demo_routing():
    print("DEMO 2 - instradamento (attenzione): un gate fa passare un input, sopprime l'altro")
    import math
    bb = Blackboard()
    hits_A = 0; hits_B = 0
    out_trace = []
    for t in range(120):
        inA = math.sin(t * 0.3)              # sorgente A
        inB = math.cos(t * 0.5)              # sorgente B
        # il task lento commuta il gate a meta'
        bb.gateSelect = 0 if t < 60 else 1
        # inibizione-gate: il selezionato passa, l'altro e' azzerato (shunting)
        gA = 1.0 if bb.gateSelect == 0 else 0.0
        gB = 1.0 if bb.gateSelect == 1 else 0.0
        out = gA * inA + gB * inB
        out_trace.append((t, bb.gateSelect, inA, inB, out))
        # verifica: l'uscita segue il selezionato
        if bb.gateSelect == 0 and abs(out - inA) < 1e-9:
            hits_A += 1
        if bb.gateSelect == 1 and abs(out - inB) < 1e-9:
            hits_B += 1
    print("  gate su A (t<60): l'uscita segue A in %d/60 scan" % hits_A)
    print("  gate su B (t>=60): l'uscita segue B in %d/60 scan" % hits_B)
    ok = hits_A == 60 and hits_B == 60
    print("  -> un segnale di controllo instrada quale input attraversa la rete,")
    print("     sopprimendo l'altro (inibizione come scambio). TEST %s\n"
          % ("PASS" if ok else "FAIL"))
    return out_trace


def main():
    print("=" * 70)
    print("FASE 4 - neuromodulazione + instradamento (multi-task, senza Main/Rule)")
    print("=" * 70 + "\n")
    log = demo_neuromodulation()
    rout = demo_routing()

    try:
        import os, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))
        scans = [r[0] for r in log]
        ws = [r[2] for r in log]
        outs = [r[3] for r in log]
        ax1.plot(scans, outs, color="#1f77b4", lw=0.8, label="uscita")
        ax1.plot(scans, ws, color="#d62728", lw=1.5, label="peso appreso")
        ax1.axhline(1.0, color="green", ls=":", lw=0.8, label="target")
        # bande di regime
        regs = []
        prev = None
        for r in log:
            if r[1] != prev:
                regs.append((r[0], r[1])); prev = r[1]
        for i, (s, name) in enumerate(regs):
            end = regs[i + 1][0] if i + 1 < len(regs) else scans[-1]
            ax1.axvspan(s, end, alpha=0.06, color="black")
            ax1.text((s + end) / 2, 1.7, name, ha="center", fontsize=8)
        ax1.set_title("Neuromodulazione: il task lento cambia regime, i pesi restano; "
                      "il comportamento cambia")
        ax1.set_xlabel("scan veloce"); ax1.set_ylabel("valore")
        ax1.set_ylim(-0.5, 2.0); ax1.legend(fontsize=8, loc="lower right"); ax1.grid(alpha=0.3)
        # routing
        ts = [r[0] for r in rout]
        ax2.plot(ts, [r[2] for r in rout], color="#2ca02c", lw=0.7, alpha=0.5, label="input A")
        ax2.plot(ts, [r[3] for r in rout], color="#ff7f0e", lw=0.7, alpha=0.5, label="input B")
        ax2.plot(ts, [r[4] for r in rout], color="#1f77b4", lw=1.8, label="uscita instradata")
        ax2.axvline(60, color="black", ls="--", lw=0.8)
        ax2.text(30, 1.15, "gate -> A", ha="center", fontsize=9)
        ax2.text(90, 1.15, "gate -> B", ha="center", fontsize=9)
        ax2.set_title("Instradamento: il gate fa passare un input e sopprime l'altro")
        ax2.set_xlabel("scan"); ax2.set_ylabel("valore"); ax2.legend(fontsize=8, loc="lower right")
        ax2.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        out = "/mnt/user-data/outputs/neuromodulation.png"
        fig.savefig(out, dpi=130)
        print("chart -> %s" % out)
    except Exception as ex:
        print("[chart skipped: %s]" % ex)


if __name__ == "__main__":
    main()
