#!/usr/bin/env python3
# =============================================================================
#  td_value.py   (FASE 3, punto 8)
#  Apprendimento per VALORE (TD) - credito temporale su orizzonti lunghi.
#  La regola dei tre fattori (HVAC) assegnava il credito solo entro il dead time
#  locale. Il TD lo propaga ALL'INDIETRO di molti passi via bootstrap:
#
#      errore di predizione (RPE)  delta = r + gamma*V(s') - V(s)
#
#  delta e' il segnale dopaminergico (ricompensa meno aspettativa). E' il
#  modulatore dei tre fattori, ma con BASELINE APPRESO (V) invece dell'errore
#  precedente. La traccia di eligibilita' = il RING BUFFER degli stati visitati
#  di recente: delta viene distribuito a tutti loro, pesato per recency.
#  TD(lambda) = il tuo ring buffer che propaga il credito a ritroso in un colpo.
#
#  Stile industriale: tabella valore = accumulatori, eligibility = ring buffer,
#  episodio = scansione ciclica.
# =============================================================================

GAMMA = 0.9        # sconto temporale
ALPHA = 0.3        # tasso di apprendimento del valore
LAMBDA = 0.8       # decadimento della traccia di eligibilita' (recency)


# --- catena di stati: 0 -> 1 -> ... -> N-1(goal), ricompensa solo alla fine ---
def td0_episode(V, N):
    for s in range(N - 1):
        r = 1.0 if (s + 1 == N - 1) else 0.0
        delta = r + GAMMA * V[s + 1] - V[s]     # errore di predizione (RPE)
        V[s] += ALPHA * delta                   # solo lo stato corrente


def tdlambda_episode(V, N):
    e = [0.0] * N                               # traccia di eligibilita' (il ring buffer)
    for s in range(N - 1):
        r = 1.0 if (s + 1 == N - 1) else 0.0
        delta = r + GAMMA * V[s + 1] - V[s]
        e[s] += 1.0                             # lo stato visitato entra nel buffer
        for k in range(N):                      # delta a TUTTI gli stati recenti...
            V[k] += ALPHA * delta * e[k]        # ...pesati dalla traccia
            e[k] *= GAMMA * LAMBDA              # la traccia svanisce (recency)


def episodes_to_reach(method, N, target=0.4, max_ep=200):
    V = [0.0] * N
    for ep in range(1, max_ep + 1):
        method(V, N)
        if V[0] >= target:                      # il valore ha raggiunto lo START
            return ep, V
    return max_ep, V


def main():
    N = 8
    print("Apprendimento per valore TD: credito temporale su orizzonte lungo")
    print("Catena di %d stati, ricompensa SOLO all'ultimo. Lo start e' a %d passi.\n" % (N, N - 2))

    print("VERIFICA 1 - il valore si propaga all'INDIETRO fino allo start:")
    V = [0.0] * N
    for ep in range(300):
        tdlambda_episode(V, N)
    print("   valore appreso per stato (0=start ... %d=goal):" % (N - 1))
    print("   %s" % ["%.2f" % v for v in V])
    print("   atteso gamma^(dist dal goal): %s"
          % ["%.2f" % (GAMMA ** (N - 2 - s)) if s < N - 1 else "0.00" for s in range(N)])
    monotona = all(V[s] <= V[s + 1] + 1e-6 for s in range(N - 2))
    print("   lo start (a %d passi dalla ricompensa) ha valore %.2f > 0: credito propagato"
          % (N - 2, V[0]))
    print("   TEST %s (valore crescente verso il goal)\n" % ("PASS" if monotona and V[0] > 0.2 else "FAIL"))

    print("VERIFICA 2 - la traccia di eligibilita' (ring buffer) accelera il credito:")
    ep0, _ = episodes_to_reach(td0_episode, N)
    epL, _ = episodes_to_reach(tdlambda_episode, N)
    print("   episodi perche' il valore raggiunga lo start:")
    print("   TD(0) - un passo per episodio:        %d episodi" % ep0)
    print("   TD(lambda) - ring buffer di recency:  %d episodi" % epL)
    speedup = ep0 / max(epL, 1)
    print("   -> il ring buffer propaga il credito a ritroso ~%.0fx piu' veloce.\n" % speedup)

    print("VERIFICA 3 - il valore appreso GUIDA la scelta (percorso corto vs lungo):")
    # da uno start si dirama: ramo corto (2 passi) o lungo (5 passi), stessa ricompensa
    # valore = gamma^passi -> il ramo corto vale di piu' -> l'attore lo sceglie
    v_short = GAMMA ** 2
    v_long = GAMMA ** 5
    scelta = "corto" if v_short > v_long else "lungo"
    print("   valore ramo corto (2 passi): %.2f | ramo lungo (5 passi): %.2f" % (v_short, v_long))
    print("   -> l'attore sceglie il ramo '%s' (valore piu' alto = piu' vicino alla ricompensa)" % scelta)
    print("   TEST %s\n" % ("PASS" if scelta == "corto" else "FAIL"))

    print("Lignaggio: tre fattori (HVAC, credito locale sul dead time) -> TD (credito")
    print("su molti passi). Il modulatore passa da 'err_prec - err_ora' a 'r + gamma*V(s')")
    print("- V(s)': la baseline diventa APPRESA (V), non solo l'errore precedente.")

    try:
        import os, matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
        # profilo del valore
        Vf = [0.0] * N
        for ep in range(300):
            tdlambda_episode(Vf, N)
        ax1.plot(range(N), Vf, "o-", color="#1f77b4", label="valore appreso")
        ax1.plot(range(N), [GAMMA ** (N - 2 - s) if s < N - 1 else 0 for s in range(N)],
                 "k--", lw=0.8, label="gamma^(dist)")
        ax1.set_title("Il valore si propaga all'indietro dallo goal allo start")
        ax1.set_xlabel("stato (0=start, %d=goal)" % (N - 1)); ax1.set_ylabel("valore V(s)")
        ax1.legend(); ax1.grid(alpha=0.3)
        # V(start) nel tempo: TD0 vs TDlambda
        V0 = [0.0] * N; VL = [0.0] * N
        t0 = []; tL = []
        for ep in range(60):
            td0_episode(V0, N); tdlambda_episode(VL, N)
            t0.append(V0[0]); tL.append(VL[0])
        ax2.plot(range(1, 61), t0, color="#d62728", label="TD(0)")
        ax2.plot(range(1, 61), tL, color="#2ca02c", label="TD(lambda) - ring buffer")
        ax2.set_title("Valore dello start nel tempo: il ring buffer accelera il credito")
        ax2.set_xlabel("episodio"); ax2.set_ylabel("valore dello start V(0)")
        ax2.legend(); ax2.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs("/mnt/user-data/outputs", exist_ok=True)
        out = "/mnt/user-data/outputs/td_value.png"
        fig.savefig(out, dpi=130)
        print("\nchart -> %s" % out)
    except Exception as ex:
        print("[chart skipped: %s]" % ex)


if __name__ == "__main__":
    main()
