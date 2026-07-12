import jax
import jax.numpy as jnp
import csv
import os
from moqui_nn.NNFacade import NetworkTopology, create_topology

def load_real_celegans_connectome(nodes_csv_path: str, edges_csv_path: str) -> NetworkTopology:
    """
    Carica la topologia reale dal dataset (es. Netzschleuder / Cook et al 2019).
    Restituisce un NetworkTopology dimensionato dinamicamente (es. N=455 se includiamo i muscoli).
    """
    # 1. Parsing dei Nodi per determinare N
    node_indices = []
    node_names = {}
    node_types = {}
    
    with open(nodes_csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if not row or row[0].startswith('#'): continue
            idx = int(row[0])
            node_indices.append(idx)
            node_types[idx] = row[1]
            node_names[idx] = row[3]
            
    N = max(node_indices) + 1
    BRANCHES = 4
    
    print(f"[CElegans] Caricamento Connettoma con {N} Nodi (Neuroni + Organi Terminali).")
    
    # 2. Inizializzazione Topologia vuota
    topo = create_topology(N, BRANCHES)
    
    # Creiamo array numpy normali per il caricamento, poi li passiamo a jax
    import numpy as np
    w_exc_np = np.zeros((N, N), dtype=np.float32)
    plasticity_np = np.zeros((N, N), dtype=bool)
    
    # 3. Parsing degli Archi (edges)
    with open(edges_csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) # skip header
        for row in reader:
            if not row or row[0].startswith('#'): continue
            source = int(row[0])
            target = int(row[1])
            weight = float(row[2]) # numero di sinapsi (usato come peso base)
            
            # Moltiplichiamo il peso per un Quantum fisiologico
            # Nel mondo PLC un peso di 1000 supera la soglia. 
            # Diamo a una singola sinapsi biologica un peso di 200.0
            w_exc_np[source, target] += weight * 200.0
            
            # Decidiamo se abilitare la plasticità
            # Abilitiamo la plasticità se la sorgente è un Interneurone
            if node_types.get(source) == 'INTERNEURONS':
                plasticity_np[source, target] = True

    # Aggiorniamo la topologia JAX
    topo = topo.replace(
        w_exc=jnp.array(w_exc_np),
        plasticity_mask=jnp.array(plasticity_np)
    )
    
    return topo, node_names, node_types
