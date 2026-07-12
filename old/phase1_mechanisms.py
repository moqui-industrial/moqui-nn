#!/usr/bin/env python3
# =============================================================================
#  phase1_mechanisms.py
#  FASE 1 della roadmap: quattro meccanismi biologici assemblati dal solo
#  primitivo (Actuator + coda di Little). Ognuno ha un TEST COSTRUTTIVO che
#  passa o fallisce - niente descrizione, solo assemblaggio verificato.
#
#   1. Plasticita' a breve termine (STP)  = deplezione della coda vescicolare
#   2. Bilancio Eccitazione-Inibizione    = inibizione che insegue l'eccitazione
#   3. Normalizzazione divisiva           = inibizione al DENOMINATORE
#   4. Dendrite non lineare               = albero di code con soglia locale
# =============================================================================


# =============================================================================
#  1. PLASTICITA' A BREVE TERMINE (STP) - Tsodyks-Markram
#     NON e' apprendimento: e' una SECONDA coda (vescicole disponibili) che si
#     svuota a ogni spike e si ricarica. I pesi appresi non cambiano.
# =============================================================================
def test_stp():
    U = 0.5           # frazione di risorse rilasciata per spike
    TAU_REC = 15.0    # costante di ricarica (scan)
    R = 1.0           # coda delle vescicole disponibili (piena)
    released = []
    # treno ad alta frequenza (spike ogni 3 scan) poi pausa poi probe
    spikes = [0] * 60
    for t in range(0, 24, 3):     # treno fitto
        spikes[t] = 1
    spikes[50] = 1                # probe dopo lunga pausa
    for t in range(60):
        # ricarica continua della coda vescicolare
        R += (1.0 - R) / TAU_REC
        if spikes[t]:
            rel = U * R           # rilascio = utilizzo x disponibile
            R -= rel              # deplezione della coda
            released.append((t, rel))
    train = [r for (t, r) in released if t < 24]
    probe = [r for (t, r) in released if t >= 50][0]
    depressione = train[0] > train[-1] * 1.3     # il treno deprime
    recupero = probe > train[-1] * 1.3           # dopo pausa recupera
    print("1. STP (deplezione coda vescicolare):")
    print("   rilascio nel treno: %s" % ["%.2f" % r for r in train])
    print("   rilascio dopo pausa: %.2f" % probe)
    print("   depressione durante il treno: %s | recupero dopo pausa: %s"
          % (depressione, recupero))
    print("   TEST %s (i pesi appresi non sono cambiati: U resta %.2f)\n"
          % ("PASS" if depressione and recupero else "FAIL", U))
    return train, [r for (t, r) in released]


# =============================================================================
#  2. BILANCIO E-I - l'inibizione insegue l'eccitazione e la bilancia.
#     Senza: l'eccitazione ricorrente esplode/satura. Con: regime controllato.
# =============================================================================
def settle(g, Wee, Wei, Wie, Wii, steps=200):
    E = 0.0; I = 0.0
    for _ in range(steps):
        dE = -E + max(0.0, g + Wee * E - Wei * I)
        dI = -I + max(0.0, Wie * E - Wii * I)
        E += 0.1 * dE; I += 0.1 * dI
        if E > 1e6: return 1e6, I           # runaway
    return E, I


def test_ei_balance():
    gs = [1, 2, 4, 6, 8, 10]
    print("2. Bilancio Eccitazione-Inibizione:")
    print("   input g | E (solo eccit.) | E (bilanciato E-I) | I insegue E?")
    bal_E = []; only_E = []; Is = []
    for g in gs:
        e_only, _ = settle(g, Wee=1.2, Wei=0.0, Wie=0.0, Wii=0.0)   # niente I
        e_bal, i_bal = settle(g, Wee=2.0, Wei=1.1, Wie=2.0, Wii=0.8)  # E-I
        only_E.append(e_only); bal_E.append(e_bal); Is.append(i_bal)
        print("     %2d    |    %8.1f     |      %6.2f        |  I=%.2f"
              % (g, e_only, e_bal, i_bal))
    runaway = only_E[-1] > 5 * bal_E[-1]             # solo-E cresce incontrollato vs bilanciato
    controlled = bal_E[-1] < 50 and bal_E[-1] > bal_E[0]   # bilanciato: controllato e crescente
    tracks = Is[-1] > Is[0]                          # I insegue E
    print("   solo-E esplode: %s | E-I controllato e monotono: %s | I insegue E: %s"
          % (runaway, controlled, tracks))
    print("   TEST %s\n" % ("PASS" if runaway and controlled and tracks else "FAIL"))
    return gs, only_E, bal_E


# =============================================================================
#  3. NORMALIZZAZIONE DIVISIVA - l'inibizione al DENOMINATORE (shunting).
#     risposta = L / (k + somma del pool) -> invarianza al contrasto/scala.
# =============================================================================
def test_divisive():
    pool_raw = [3.0, 1.0, 0.5, 0.2]     # attivita' grezze di 4 neuroni
    K = 1.0
    print("3. Normalizzazione divisiva (invarianza al contrasto):")
    print("   contrasto | risposte grezze        | risposte normalizzate")
    norms = []
    for c in [1.0, 3.0, 9.0]:
        raw = [r * c for r in pool_raw]
        s = sum(raw)
        norm = [r / (K + s) for r in raw]
        norms.append(norm)
        print("     x%.0f     | %s | %s"
              % (c, ["%.1f" % r for r in raw], ["%.2f" % n for n in norm]))
    # invarianza: le risposte normalizzate quasi non cambiano col contrasto
    import math
    drift = max(abs(norms[2][i] - norms[0][i]) for i in range(4))
    invariante = drift < 0.15
    print("   deriva max della risposta normalizzata su contrasto x1->x9: %.3f" % drift)
    print("   TEST %s (grezze scalano col contrasto, normalizzate no)\n"
          % ("PASS" if invariante else "FAIL"))
    return norms


# =============================================================================
#  4. DENDRITE NON LINEARE - albero di code con SOGLIA LOCALE (NMDA spike).
#     Due input sullo STESSO ramo -> coincidenza locale amplificata.
#     Due input su rami DIVERSI    -> somma lineare, niente boost.
#     Il neurone calcola un OR-di-AND che il soma lineare non puo' fare.
# =============================================================================
def branch_out(inp, th=1.5, boost=3.0):
    # NMDA: se l'ingresso del ramo supera la soglia locale, scatta e amplifica
    return inp + boost if inp > th else inp


def soma_fires(branch_inputs, soma_th=3.0):
    return sum(branch_out(b) for b in branch_inputs) >= soma_th


def test_nonlinear_dendrite():
    # 3 rami. Ogni input singolo vale 1.0.
    same_branch = [2.0, 0.0, 0.0]     # due input sullo stesso ramo
    diff_branch = [1.0, 1.0, 0.0]     # due input su rami diversi
    fires_same = soma_fires(same_branch)
    fires_diff = soma_fires(diff_branch)
    # un neurone puntiforme (somma poi soglia) vedrebbe 2.0 in entrambi i casi
    point_same = sum(same_branch) >= 3.0
    point_diff = sum(diff_branch) >= 3.0
    print("4. Dendrite non lineare (albero di code con soglia locale):")
    print("   due input STESSO ramo -> soma spara: %s (uscita rami %.1f)"
          % (fires_same, sum(branch_out(b) for b in same_branch)))
    print("   due input rami DIVERSI -> soma spara: %s (uscita rami %.1f)"
          % (fires_diff, sum(branch_out(b) for b in diff_branch)))
    print("   neurone puntiforme (somma sola): stesso=%s diversi=%s -> NON distingue"
          % (point_same, point_diff))
    ok = fires_same and not fires_diff and (point_same == point_diff)
    print("   TEST %s (il dendrite-albero calcola cio' che il soma lineare non fa)\n"
          % ("PASS" if ok else "FAIL"))


def main():
    print("=" * 68)
    print("FASE 1 - quattro meccanismi assemblati dal primitivo (Actuator+coda)")
    print("=" * 68 + "\n")
    train, stp_full = test_stp()
    gs, only_E, bal_E = test_ei_balance()
    norms = test_divisive()
    test_nonlinear_dendrite()

    try:
        import os, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 2, figsize=(13, 9))
        # 1 STP
        axs[0, 0].plot(range(len(stp_full)), stp_full, "o-", color="#1f77b4")
        axs[0, 0].set_title("1. STP: rilascio deprime nel treno, recupera dopo la pausa")
        axs[0, 0].set_xlabel("spike n."); axs[0, 0].set_ylabel("rilascio"); axs[0, 0].grid(alpha=0.3)
        # 2 E-I
        axs[0, 1].plot(gs, bal_E, "o-", color="#2ca02c", label="bilanciato E-I")
        axs[0, 1].plot(gs, [min(e, 60) for e in only_E], "o--", color="#d62728", label="solo E (esplode)")
        axs[0, 1].set_title("2. Bilancio E-I: controllato vs runaway")
        axs[0, 1].set_xlabel("input g"); axs[0, 1].set_ylabel("attivita' E")
        axs[0, 1].set_ylim(0, 60); axs[0, 1].legend(); axs[0, 1].grid(alpha=0.3)
        # 3 divisive
        x = range(4)
        for c, norm, col in zip([1, 3, 9], norms, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            axs[1, 0].plot(x, norm, "o-", color=col, label="contrasto x%d" % c)
        axs[1, 0].set_title("3. Normalizzazione divisiva: normalizzate invarianti al contrasto")
        axs[1, 0].set_xlabel("neurone"); axs[1, 0].set_ylabel("risposta normalizzata")
        axs[1, 0].legend(); axs[1, 0].grid(alpha=0.3)
        # 4 dendrite
        cats = ["stesso ramo", "rami diversi"]
        vals = [sum(branch_out(b) for b in [2.0, 0, 0]), sum(branch_out(b) for b in [1.0, 1.0, 0])]
        bars = axs[1, 1].bar(cats, vals, color=["#2ca02c", "#d62728"])
        axs[1, 1].axhline(3.0, color="black", ls="--", label="soglia soma")
        axs[1, 1].set_title("4. Dendrite: coincidenza locale supera la soglia, sparsa no")
        axs[1, 1].set_ylabel("uscita totale rami"); axs[1, 1].legend()
        fig.tight_layout()
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        out = "/mnt/user-data/outputs/phase1_mechanisms.png"
        fig.savefig(out, dpi=130)
        print("chart -> %s" % out)
    except Exception as ex:
        print("[chart skipped: %s]" % ex)


if __name__ == "__main__":
    main()
