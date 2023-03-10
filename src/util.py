from itertools import product
from pathlib import Path
from typing import Optional, Tuple

# import jax.numpy as jnp
import numpy as np


class WelfordEstimator:
    """Compute running mean and standard deviations.
    The Welford approach greatly reduces memory consumption.
    """

    def __init__(self) -> None:
        """Create a Welfordestimator."""
        self.collapsed_axis: Optional[Tuple[int, ...]] = None

    # estimate running mean and std
    # average all axis except the color channel
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    def update(self, batch_vals: np.ndarray) -> None:
        """Update the running estimation.
        Args:
            batch_vals (np.ndarray): The current batch element.
        """
        if not self.collapsed_axis:
            self.collapsed_axis = tuple(np.arange(len(batch_vals.shape[:-1])))
            self.count = np.zeros(1, dtype=np.float64)
            self.mean = np.zeros(batch_vals.shape[-1], dtype=np.float64)
            self.std = np.zeros(batch_vals.shape[-1], dtype=np.float64)
            self.m2 = np.zeros(batch_vals.shape[-1], dtype=np.float64)
        self.count += np.prod(np.array(batch_vals.shape[:-1]))
        delta = np.subtract(batch_vals, self.mean)
        self.mean += np.sum(delta / self.count, self.collapsed_axis)
        delta2 = np.subtract(batch_vals, self.mean)
        self.m2 += np.sum(delta * delta2, self.collapsed_axis)

    def finalize(self) -> Tuple[np.ndarray, np.ndarray]:
        """Finish the estimation and return the computed mean and std.
        Returns:
            Tuple[np.ndarray, np.ndarray]: Estimated mean and variance.
        """
        return self.mean, np.sqrt(self.m2 / self.count)


def get_label_of_folder(
    path_of_folder: Path, binary_classification: bool = False
) -> int:
    """Get the label of the images in a folder based on the folder path.
        We assume:
            A: Orignal data, B: First gan,
            C: Second gan, D: Third gan, E: Fourth gan.
        A working folder structure could look like:
            A_celeba  B_CramerGAN  C_MMDGAN  D_ProGAN  E_SNGAN
        With each folder containing the images from the corresponding
        source.
    Args:
        path_of_folder (Path):  Path string containing only a single
            underscore directly after the label letter.
        binary_classification (bool): If flag is set, we only classify binarily, i.e. whether an image is real or fake.
            In this case, the prefix 'A' indicates real, \
            which is encoded with the label 0. All other folders are considered
            fake data, encoded with the label 1.
    Raises:
       NotImplementedError: Raised if the label letter is unkown.
    Returns:
        int: The label encoded as integer.
    # noqa: DAR401
    """
    label_str = path_of_folder.name.split("_")[0]
    if binary_classification:
        # differentiate original and generated data
        if label_str == "A":
            return 0
        else:
            return 1
    else:
        # the the label based on the path, As are 0s, Bs are 1, etc.
        if label_str == "A":
            label = 0
        elif label_str == "B":
            label = 1
        elif label_str == "C":
            label = 2
        elif label_str == "D":
            label = 3
        elif label_str == "E":
            label = 4
        else:
            raise NotImplementedError(label_str)
        return label


def get_label(path_to_image: Path, binary_classification: bool) -> int:
    """Get the label based on the image path.
       The file structure must be as outlined in the README file.
       We assume:
            A: Orignal data, B: First gan,
            C: Second gan, D: Third gan, E: Fourth gan.
       A working folder structure could look like:
            A_celeba  B_CramerGAN  C_MMDGAN  D_ProGAN  E_SNGAN
       With each folder containing the images from the corresponding source.
    Args:
        path_to_image (Path): Image path string containing only a single
            underscore directly after the label letter.
        binary_classification (bool): If flag is set, we only classify binarily, i.e. whether an image is real or fake.
            In this case, the prefix 'A' indicates real, which is encoded with the label 0.
            All other folders are considered fake data, encoded with the label 1.
    Raises:
        NotImplementedError: Raised if the label letter is unkown.
    Returns:
        int: The label encoded as integer.
    """
    return get_label_of_folder(path_to_image.parent, binary_classification)


def load_folder(
    folder: Path, train_size: int, val_size: int, test_size: int
) -> np.ndarray:
    """Create posix-path lists for png files in a folder.
    Given a folder containing portable network graphics (*.png) files
    this functions will create Posix-path lists. A train, test, and
    validation set list is created.
    Args:
        folder: Path to a folder with images from the same source, i.e. A_ffhq .
        train_size: Desired size of the training set.
        val_size: Desired size of the validation set.
        test_size: Desired size of the test set.
    Returns:
        Numpy array with the train, validation and test lists, in this order.
    Raises:
        ValueError: if the requested set sizes are not smaller or equal to the number of images available

    # noqa: DAR401
    """
    file_list = list(folder.glob("./*.png"))
    if len(file_list) < train_size + val_size + test_size:
        raise ValueError(
            "Requested set sizes must be smaller or equal to the number of images available."
        )

    # split the list into training, validation and test sub-lists.
    train_list = file_list[:train_size]
    validation_list = file_list[train_size : (train_size + val_size)]
    test_list = file_list[(train_size + val_size) : (train_size + val_size + test_size)]
    return np.asarray([train_list, validation_list, test_list], dtype=object)
