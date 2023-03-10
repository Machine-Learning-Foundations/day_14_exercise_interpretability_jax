import pickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt


from flax import linen as nn


class CNN(nn.Module):
    """A CNN model."""

    @nn.compact
    def __call__(self, x):
        """Run the forward pass."""
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        return nn.sigmoid(x)


if __name__ == '__main__':

    weights = pickle.load(open('./data/weights.pkl', 'rb'))
    net = CNN()
    neuron = 6

    def forward_pass(x):
        out = net.apply(weights, x)
        return out[0, neuron]

    get_grads = jax.value_and_grad(forward_pass)

    x = jax.random.uniform(jax.random.PRNGKey(42),
            [1, 28, 28, 1])
    x = jnp.ones([1, 28, 28, 1])

    grads = []
    for i in range(25):
        x = (x - jnp.mean(x))/jnp.std(x + 1e-5)
        act, grad = get_grads(x)
        x = x + grad
        grads.append(grad)
        print(act)

    mean_grad = jnp.mean(jnp.stack(grads, 0), 0)

    plt.imshow(x[0, :, :, 0])
    plt.title('Input maxizing the ' + str(neuron) + '- neuron')
    plt.show()
