import jax
import jax.numpy as jnp
from flax import linen as nn
import optax
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import time

class SimpleANN(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Flatten implicito
        x = nn.Dense(features=10)(x)
        return x

def create_train_state(rng, learning_rate):
    model = SimpleANN()
    params = model.init(rng, jnp.ones((1, 64)))['params']
    tx = optax.adam(learning_rate)
    from flax.training.train_state import TrainState
    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, batch_images, batch_labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch_images)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch_labels).mean()
        return loss, logits
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch_labels)
    return state, accuracy

def test_mnist_ann():
    print("--- Artificial Neural Network (Flax) su Scikit-Learn Digits ---")
    digits = load_digits()
    # Normalize a 0-1
    X = digits.data / 16.0 
    y = digits.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, learning_rate=0.01)
    
    epochs = 50
    batch_size = 32
    
    start_time = time.time()
    for epoch in range(epochs):
        accs = []
        for i in range(0, len(X_train), batch_size):
            batch_x = jnp.array(X_train[i:i+batch_size])
            batch_y = jnp.array(y_train[i:i+batch_size])
            state, acc = train_step(state, batch_x, batch_y)
            accs.append(acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Accuracy {jnp.mean(jnp.array(accs)):.4f}")
            
    train_time = time.time() - start_time
    
    # Test
    logits = state.apply_fn({'params': state.params}, jnp.array(X_test))
    preds = jnp.argmax(logits, -1)
    test_acc = jnp.mean(preds == jnp.array(y_test))
    
    print(f"--- Risultati ANN ---")
    print(f"Tempo di training: {train_time:.2f} secondi (per {epochs} epoche)")
    print(f"Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    test_mnist_ann()
