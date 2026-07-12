import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class LogRingBuffer:
    """
    LoggerFacade: Registratore circolare ad alte prestazioni delle Status Word (PLC).
    Shape: [LOG_SIZE, N_neurons]
    """
    status_words: jnp.ndarray
    head: jnp.int32
    tail: jnp.int32
    size: jnp.int32
    max_size: jnp.int32

def init_log_ring_buffer(max_size: int, n_neurons: int) -> LogRingBuffer:
    return LogRingBuffer(
        status_words=jnp.zeros((max_size, n_neurons), dtype=jnp.uint8),
        head=jnp.int32(0),
        tail=jnp.int32(0),
        size=jnp.int32(0),
        max_size=jnp.int32(max_size)
    )

def log_push(buffer: LogRingBuffer, current_status_word: jnp.ndarray) -> LogRingBuffer:
    """
    Scrive la Status Word (vettore di stati FSM di tutti i neuroni) nell'indice corrente.
    """
    new_status_words = buffer.status_words.at[buffer.tail].set(current_status_word)
    
    new_tail = (buffer.tail + 1) % buffer.max_size
    
    # Se il buffer è pieno, la testa avanza sovrascrivendo i dati vecchi
    new_head = jnp.where(buffer.size >= buffer.max_size, (buffer.head + 1) % buffer.max_size, buffer.head)
    new_size = jnp.where(buffer.size < buffer.max_size, buffer.size + 1, buffer.size)
    
    return LogRingBuffer(
        status_words=new_status_words,
        head=new_head,
        tail=new_tail,
        size=new_size,
        max_size=buffer.max_size
    )

def extract_log(buffer: LogRingBuffer) -> jnp.ndarray:
    """
    Estrae il log linearizzato dal più vecchio al più recente.
    Dato che JAX richiede dimensioni statiche in uscita, restituisce sempre un tensore di shape [MAX_SIZE, N],
    ruotando l'array in modo che l'evento più vecchio sia all'indice 0.
    """
    # jnp.roll sposta gli elementi per far combaciare head a 0
    shifted_log = jnp.roll(buffer.status_words, shift=-buffer.head, axis=0)
    return shifted_log
