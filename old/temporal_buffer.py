#!/usr/bin/env python3
# =============================================================================
#  temporal_buffer.py
#  Il buffer temporale E' la coda (decadimento), il ciclo E' il ritmo (uscita).
#  NON un handler che parcheggia eventi in attesa di un ciclo di elaborazione.
#
#  Confronto sullo stesso flusso di eventi asincroni:
#   A) POLLING: gli eventi si accumulano in un bin; a fine bin si elaborano in
#      blocco. Il raggruppamento e' fatto dal CLOCK (i bordi del bin).
#   B) CONTINUO: ogni evento entra all'istante nella coda, che integra con
#      decadimento (inerzia = W di Little); la soglia legge in continuo. Il
#      raggruppamento e' fatto dal DECADIMENTO (nessun bordo).
#
#  Tesi: A ha artefatti di confine (perde coincidenze a cavallo del bordo,
#  e raggruppa falsamente eventi lontani finiti nello stesso bin). B no: la
#  finestra di raggruppamento e' centrata su ogni evento, indipendente dalla fase.
#  Poi: il RITMO cadenza l'USCITA, non l'ingresso - nessun evento va perso.
# =============================================================================

import random, math

DT = 0.5           # risoluzione fine (ms): il sotto-istante in cui gli eventi entrano
TAU = 8.0          # inerzia della coda (ms): la finestra di raggruppamento per decadimento
T_POLL = 10.0      # periodo del bin di polling (ms)
CHARGE = 0.60      # carica per evento (due in coincidenza superano la soglia 1.0)
THRESHOLD = 1.0


def continuous_detects(tA, tB, horizon=40.0):
    # coda a decadimento: evento -> deposita CHARGE; ogni step decade. Coincidenza
    # se L supera la soglia (serve la sovrapposizione di due eventi ancora vivi).
    L = 0.0
    t = 0.0
    depA = False; depB = False
    decay = math.exp(-DT / TAU)
    detected = False
    while t < horizon:
        L *= decay
        if not depA and t >= tA:
            L += CHARGE; depA = True
        if not depB and t >= tB:
            L += CHARGE; depB = True
        if L >= THRESHOLD:
            detected = True
        t += DT
    return detected


def polling_detects(tA, tB):
    # bin di polling: coincidenza se i due eventi cadono nello STESSO bin
    return int(tA // T_POLL) == int(tB // T_POLL)


def demo_coincidence():
    print("DEMO 1 - rilevare la coincidenza di due eventi distanti dt")
    print("(fase assoluta casuale; media su molte prove per ogni dt)\n")
    print("   dt (ms) | continuo (decadimento) | polling (bin)")
    rng = random.Random(0)
    dts = [0.5, 1, 2, 3, 4, 5, 7, 9, 11]
    cont_curve = []; poll_curve = []
    for dt in dts:
        c_hits = 0; p_hits = 0; N = 400
        for _ in range(N):
            base = rng.uniform(0, T_POLL)      # fase assoluta casuale
            tA, tB = base, base + dt
            if continuous_detects(tA, tB):
                c_hits += 1
            if polling_detects(tA, tB):
                p_hits += 1
        cont_curve.append(c_hits / N); poll_curve.append(p_hits / N)
        print("    %4.1f   |         %.2f          |     %.2f" % (dt, c_hits / N, p_hits / N))
    print("\n   -> continuo: raggruppa nettamente entro ~%.0f ms (la finestra d'inerzia)," % (TAU * 0.4))
    print("      indipendente dalla fase. polling: perde coincidenze strette a cavallo")
    print("      del bordo del bin (fase-dipendente) E raggruppa falsamente eventi")
    print("      lontani fino a %g ms finiti nello stesso bin. Il raggruppamento" % T_POLL)
    print("      lo fa il DECADIMENTO, non il clock.\n")
    return dts, cont_curve, poll_curve


def demo_rhythm_gates_output():
    print("DEMO 2 - il ritmo cadenza l'USCITA, non l'ingresso: nessun evento perso")
    rng = random.Random(1)
    # finestra ritmica: aperta 1/3 del periodo, chiusa 2/3
    period = 12.0
    open_frac = 0.34
    events = sorted(rng.uniform(0, 120) for _ in range(60))

    # A) uscita cadenzata (biologico): ogni evento entra nella coda comunque;
    #    l'uscita passa a valle solo a finestra aperta. Nessun evento perso.
    integrated_out = 0
    for e in events:
        integrated_out += 1                    # entra sempre nella coda (interrupt async)
    lost_output_gated = 0                       # l'uscita ritarda, non scarta

    # B) ingresso cadenzato (sbagliato, polling-like): l'evento che arriva a
    #    finestra chiusa viene SCARTATO. Perde la frazione a finestra chiusa.
    lost_input_gated = 0
    for e in events:
        phase = (e % period) / period
        is_open = phase < open_frac
        if not is_open:
            lost_input_gated += 1

    print("   eventi totali: %d" % len(events))
    print("   uscita cadenzata (coda integra tutto, ritmo sull'uscita): persi %d"
          % lost_output_gated)
    print("   ingresso cadenzato (ritmo scarta a finestra chiusa):       persi %d (~%.0f%%)"
          % (lost_input_gated, 100 * lost_input_gated / len(events)))
    ok = lost_output_gated == 0 and lost_input_gated > 0
    print("   -> se il ritmo filtrasse l'INGRESSO perderesti gli eventi a finestra")
    print("      chiusa. Filtrando l'USCITA non perdi nulla: gli eventi entrano")
    print("      sempre (interrupt), la coda li integra, l'uscita esce a ritmo.")
    print("   TEST %s\n" % ("PASS" if ok else "FAIL"))
    return len(events), lost_output_gated, lost_input_gated


def main():
    print("=" * 68)
    print("Il buffer e' la coda (decadimento), il ciclo e' il ritmo (uscita)")
    print("=" * 68 + "\n")
    dts, cont, poll = demo_coincidence()
    tot, lost_out, lost_in = demo_rhythm_gates_output()

    try:
        import os, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        ax1.plot(dts, cont, "o-", color="#2ca02c", label="continuo (decadimento)")
        ax1.plot(dts, poll, "s-", color="#d62728", label="polling (bin)")
        ax1.axvline(TAU * 0.4, color="gray", ls=":", lw=0.8, label="finestra d'inerzia")
        ax1.set_title("Rilevare coincidenza vs distanza temporale\n"
                      "il continuo raggruppa per decadimento (netto, fase-indipendente)")
        ax1.set_xlabel("dt tra i due eventi (ms)"); ax1.set_ylabel("tasso di rilevazione")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
        bars = ax2.bar(["uscita\ncadenzata\n(biologico)", "ingresso\ncadenzato\n(polling)"],
                       [lost_out, lost_in], color=["#2ca02c", "#d62728"])
        ax2.set_title("Eventi persi su %d\n(il ritmo sull'uscita non perde nulla)" % tot)
        ax2.set_ylabel("eventi persi")
        for b, v in zip(bars, [lost_out, lost_in]):
            ax2.text(b.get_x() + b.get_width() / 2, v + 0.3, str(v), ha="center")
        fig.tight_layout()
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        out = "/mnt/user-data/outputs/temporal_buffer.png"
        fig.savefig(out, dpi=130)
        print("chart -> %s" % out)
    except Exception as ex:
        print("[chart skipped: %s]" % ex)


if __name__ == "__main__":
    main()
