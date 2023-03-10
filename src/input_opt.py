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

    ## TODO: Find an approximation of the input that maximizes a
    # specific output neuron.

