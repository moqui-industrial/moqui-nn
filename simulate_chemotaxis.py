import jax
import jax.numpy as jnp
import os

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_PLT = True
except ImportError:
    HAVE_PLT = False

from moqui_nn.CElegans import load_real_celegans_connectome
from moqui_nn.DeviceManager import init_network_state, network_step
from moqui_nn.Environment import init_environment, environment_step
from moqui_nn.IOFacade import get_celegans_io_indices, map_sensory_inputs, map_motor_outputs
from moqui_nn.Main import Blackboard, ecosystem_update

def simulate_chemotaxis():
    print("--- C. Elegans Chemotaxis Simulation (JAX) ---")
    
    # 1. Caricamento Hardware
    nodes_csv = "datasets/nodes.csv"
    edges_csv = "datasets/edges.csv"
    
    if not os.path.exists(nodes_csv):
        print("Dataset non trovato! Assicurarsi di aver scaricato il dataset di Netzschleuder.")
        return
        
    topo, node_names, node_types = load_real_celegans_connectome(nodes_csv, edges_csv)
    
    # 2. Indici IO
    sens_idx, fwd_idx, left_idx, right_idx = get_celegans_io_indices(node_names)
    print(f"  Trovati Sensori: {len(sens_idx)}, Motori Avanti: {len(fwd_idx)}, Sterzo SX: {len(left_idx)}, Sterzo DX: {len(right_idx)}")
    
    # 3. Inizializzazione Stati
    MAX_DELAY = 4
    net_state = init_network_state(topo, MAX_DELAY)
    env_state = init_environment(food_x=50.0, food_y=50.0) # Il cibo si trova a (50, 50)
    blackboard = Blackboard(dopamine_level=0.1, serotonin_level=0.0, sleep_mode=False)
    
    # 4. Creazione Step Integrato per lax.scan
    def simulation_step(carry, _):
        net_state, env_state, blackboard = carry
        
        # IO Facade: Input (Ambiente -> Rete)
        ext_exc = map_sensory_inputs(env_state.last_odor, sens_idx, topo.n_neurons)
        ext_inh = jnp.zeros(topo.n_neurons, dtype=jnp.float32)
        disable_req = jnp.zeros(topo.n_neurons, dtype=jnp.bool_)
        
        # Esecuzione Scan Rete (1 tick)
        new_net_state, out_spikes = network_step(net_state, topo, blackboard, ext_exc, ext_inh, disable_req)
        
        # IO Facade: Output (Rete -> Ambiente)
        thrusts = map_motor_outputs(out_spikes, fwd_idx, left_idx, right_idx)
        forward_thrust = thrusts[0]
        turn_thrust = thrusts[1]
        
        # Aggiornamento Ambiente
        new_env_state = environment_step(env_state, forward_thrust, turn_thrust, dt=0.1)
        
        # Ecosystem Update (Reward: se ci stiamo avvicinando al cibo, aumentiamo la dopamina)
        # delta_odor positivo = avvicinamento -> reward
        delta_odor = new_env_state.last_odor - env_state.last_odor
        reward = jnp.where(delta_odor > 0, delta_odor * 100.0, -delta_odor * 10.0) # Punizione leggera per allontanamento
        new_blackboard = ecosystem_update(blackboard, reward)
        
        new_carry = (new_net_state, new_env_state, new_blackboard)
        
        # Registriamo la posizione per il grafico
        out_data = jnp.array([new_env_state.worm_x, new_env_state.worm_y, new_env_state.last_odor])
        
        return new_carry, out_data
        
    M = 2000 # Numero di step di simulazione (es. 2 secondi a 1ms)
    print(f"  Avvio compilazione JIT ed esecuzione per {M} scan...")
    
    # Esecuzione del loop
    initial_carry = (net_state, env_state, blackboard)
    
    # Usiamo jax.jit su lax.scan
    @jax.jit
    def run_sim(carry):
        return jax.lax.scan(simulation_step, carry, None, length=M)
        
    final_carry, history = run_sim(initial_carry)
    
    print("  Simulazione completata.")
    
    # Plot della traiettoria
    if HAVE_PLT:
        x_hist = history[:, 0]
        y_hist = history[:, 1]
        odor_hist = history[:, 2]
        
        os.makedirs("./outputs", exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(x_hist, y_hist, label="Worm Path", color="blue")
        ax.scatter([0.0], [0.0], color="green", label="Start", marker="o")
        ax.scatter([50.0], [50.0], color="red", label="Food", marker="*", s=200)
        ax.set_title("C. Elegans JAX Chemotaxis")
        ax.legend()
        fig.tight_layout()
        fig.savefig("./outputs/celegans_chemotaxis.png", dpi=130)
        print("  -> Grafico salvato in ./outputs/celegans_chemotaxis.png")

if __name__ == "__main__":
    simulate_chemotaxis()
