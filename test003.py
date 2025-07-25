import pennylane as qml
import jax.numpy as jnp

import jax

# Set number of wires
num_wires = 28

# Set a random seed
key = jax.random.PRNGKey(0)

dev = qml.device("lightning.gpu", wires=num_wires)

@qml.qjit(autograph=True)
@qml.qnode(dev)
def circuit(params):

    # Apply layers of RZ and RY rotations
    for i in range(num_wires):
        qml.RZ(params[3*i], wires=[i])
        qml.RY(params[3*i+1], wires=[i])
        qml.RZ(params[3*i+2], wires=[i])

    return qml.expval(qml.PauliZ(0) + qml.PauliZ(num_wires-1))

# Initialize the weights
weights = jax.random.uniform(key, shape=(3 * num_wires,), dtype=jnp.float32)

@qml.qjit(autograph=True)
def workflow(params):
    g = qml.grad(circuit)
    return g(params)

import catalyst
import optax

opt = optax.sgd(learning_rate=0.4)

def update_step(i, params, opt_state):
    energy, grads = catalyst.value_and_grad(circuit)(params)
    updates, opt_state = opt.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    catalyst.debug.print("Step = {i}", i=i)
    return (params, opt_state)

@qml.qjit
def optimization(params):
    opt_state = opt.init(params)
    (params, opt_state) = qml.for_loop(0, 10, 1)(update_step)(params, opt_state)
    return params


