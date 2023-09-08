"""Train deepfake detector on SytleGAN deepfakes."""
from multiprocessing import Pool
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax  # Optimizers
from flax import linen as nn  # The Linen API
from flax.training import train_state  # Useful dataclass to keep train state
from PIL import Image
from tqdm import tqdm

from mnist_integrated import integrate_gradients
from util import WelfordEstimator, get_label, load_folder


def load_image(path_to_file: Path) -> np.ndarray:
    """Load image from path."""
    image = Image.open(path_to_file)
    array = np.nan_to_num(np.array(image), posinf=255, neginf=0)
    return array


class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        """Forward step."""
        x = nn.Conv(features=16, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=128, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=8192)(x)
        x = nn.relu(x)
        x = nn.Dense(features=2)(x)
        return x


class Dense(nn.Module):
    """A simple Dense model."""

    @nn.compact
    def __call__(self, x):
        """Forward step."""
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=2)(x)
        return x


@jax.jit
def cross_entropy_loss(*, logits, labels):
    """Calculate cross entropy loss."""
    labels_onehot = jax.nn.one_hot(labels, num_classes=2)
    return optax.softmax_cross_entropy(logits=logits, labels=labels_onehot).mean()


def create_train_state(rng, learning_rate):
    """Create initial `TrainState`."""
    params = net.init(rng, jnp.ones([1, 128, 128, 3]))["params"]
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)


def compute_metrics(*, logits, labels):
    """Compute metrics after training step."""
    loss = cross_entropy_loss(logits=logits, labels=labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": loss,
        "accuracy": accuracy,
    }
    return metrics


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        """Define loss function."""
        logits = net.apply({"params": params}, batch["image"])
        loss = cross_entropy_loss(logits=logits, labels=batch["label"])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    metrics = compute_metrics(logits=logits, labels=batch["label"])
    return state, metrics


@jax.jit
def eval_step(params, batch):
    """Make one eval step."""
    logits = net.apply({"params": params}, batch["image"])
    return compute_metrics(logits=logits, labels=batch["label"])


def transform(image_data):
    """Transform image data."""
    # TODO: Implement the function given in the readme
    return jnp.zeros_like(0)


if __name__ == "__main__":
    ffhq_train, ffhq_val, ffhq_test = load_folder(
        Path("./data/ffhq_style_gan/source_data/A_ffhq"), 67_000, 1_000, 2_000
    )
    gan_train, gan_val, gan_test = load_folder(
        Path("./data/ffhq_style_gan/source_data/B_stylegan"), 67_000, 1_000, 2_000
    )

    fft = True
    epochs = 5
    batch_size = 500

    train = np.concatenate((ffhq_train, gan_train))
    val = np.concatenate((ffhq_val, gan_val))
    np.random.seed(42)
    rng = jax.random.PRNGKey(0)

    np.random.shuffle(train)
    np.random.shuffle(val)

    train_batches = np.array_split(train, len(train) // batch_size)[:50]

    # net = CNN()
    net = Dense()
    state = create_train_state(rng, 0.001)

    estimator = WelfordEstimator()
    image_batch_list = []
    label_batch_list = []

    with Pool(5) as p:
        for path_batch in tqdm(train_batches, "computing training mean and std"):
            loaded = np.stack(p.map(load_image, path_batch))
            image_stack = np.stack(loaded)
            image_batch_list.append(image_stack)
            label_batch_list.append(
                np.array([get_label(path, True) for path in path_batch])
            )
            if fft:
                transform_batch = transform(image_stack)
                estimator.update(transform_batch)
            else:
                estimator.update(image_stack)
    train_mean, train_std = estimator.finalize()
    train_mean, train_std = train_mean.astype(np.float32), train_std.astype(np.float32)
    train_mean, train_std = jnp.array(train_mean), jnp.array(train_std)

    print("mean: {}, std: {}".format(train_mean, train_std))

    val_image = jnp.stack(list(map(load_image, val)))
    if fft:
        transform_val = transform(val_image)
        val_image = transform_val
    val_label = jnp.array([get_label(path, True) for path in val])
    val_ds = {}
    val_ds["image"] = (val_image - train_mean) / train_std
    val_ds["label"] = val_label

    for e in range(epochs):
        metrics = eval_step(state.params, val_ds)
        print(
            "val  , epoch {}, loss {:3.3f}, acc {:3.3f}".format(
                e, metrics["loss"], metrics["accuracy"]
            )
        )

        progress_bar = tqdm(
            zip(image_batch_list, label_batch_list), total=len(image_batch_list)
        )
        for img_batch, label_batch in progress_bar:
            if fft:
                img_batch = transform(img_batch)
            img_batch = (img_batch - train_mean) / train_std

            train_ds = {}
            train_ds["image"] = img_batch
            train_ds["label"] = label_batch

            state, metrics = train_step(state, train_ds)
            progress_bar.set_description(
                "Training. Loss: {:3.3f}, Acc: {:3.3f}".format(
                    metrics["loss"], metrics["accuracy"]
                )
            )

    metrics = eval_step(state.params, val_ds)
    print(
        "val  , epoch {}, loss {:3.3f}, acc {:3.3f}".format(
            e, metrics["loss"], metrics["accuracy"]
        )
    )

    # test metrics
    test = np.concatenate((ffhq_test, gan_test))
    # load test data
    test_image = jnp.stack(list(map(load_image, test)))
    if fft:
        test_image = transform(test_image)
    test_label = jnp.array([get_label(path, True) for path in test])
    test_ds = {}
    test_ds["image"] = (test_image - train_mean) / train_std
    test_ds["label"] = test_label
    # get the
    test_metrics = eval_step(state.params, test_ds)
    print(
        "test, loss {:3.3f}, acc {:3.3f}".format(
            test_metrics["loss"], test_metrics["accuracy"]
        )
    )

    # visualize the linear network.
    if type(net) is Dense:
        import matplotlib.pyplot as plt

        stacked_ffhq_val = jnp.stack(list(map(load_image, ffhq_val)))
        fft_ffhq_val = transform(stacked_ffhq_val)
        stacked_gan_val = jnp.stack(list(map(load_image, gan_val)))
        fft_gan_val = transform(stacked_gan_val)

        fft_ffhq_val = np.mean(fft_ffhq_val, [0, -1])
        fft_gan_val = np.mean(fft_gan_val, [0, -1])

        diff = np.abs(fft_ffhq_val - fft_gan_val)
        plt.subplot(1, 2, 1)
        plt.title("Real mean-log fft2")
        plt.imshow(fft_ffhq_val, vmin=np.min(fft_ffhq_val), vmax=np.max(fft_ffhq_val))
        plt.colorbar()
        plt.subplot(1, 2, 2)
        plt.title("Fake mean-log fft2")
        plt.imshow(fft_gan_val, vmin=np.min(fft_ffhq_val), vmax=np.max(fft_ffhq_val))
        plt.colorbar()
        plt.savefig("real_fake_mean-log_fft2.jpg")

        plt.subplots()
        plt.title("Row averaged shifted mean-log fft2")
        plt.plot(np.fft.fftshift(np.mean(fft_ffhq_val, 0))[64:], ".", label="real")
        plt.plot(np.fft.fftshift(np.mean(fft_gan_val, 0))[64:], ".", label="fake")
        plt.xlabel("frequency")
        plt.ylabel("magnitude")
        plt.legend()
        plt.savefig("row_average_shifted_mean-log_fft2.jpg")

        plt.subplots()
        plt.title("Mean frequency difference")
        plt.imshow(diff)
        plt.colorbar()
        plt.savefig("mean_freq_difference.jpg")

        weight_ffhq = np.mean(
            np.reshape(state.params["Dense_0"]["kernel"][:, 0], (128, 128, 3)), -1
        )
        weight_style = np.mean(
            np.reshape(state.params["Dense_0"]["kernel"][:, 1], (128, 128, 3)), -1
        )

        plt.subplot(1, 2, 1)
        plt.title("Real classifier weights")
        plt.imshow(weight_ffhq, vmin=np.min(weight_ffhq), vmax=np.max(weight_ffhq))
        plt.subplot(1, 2, 2)
        plt.title("Fake classifier weights")
        plt.imshow(weight_style, vmin=np.min(weight_ffhq), vmax=np.max(weight_ffhq))
        plt.colorbar()

        plt.savefig("classifier_weights.jpg")

    if type(net) is CNN:
        import matplotlib.pyplot as plt

        ig_out = integrate_gradients(net, state.params, val_ds, 1)
        plt.imshow(np.mean(ig_out, -1))

        plt.savefig("integrated_gradients.jpg")
