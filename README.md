# Exercise: Interpretable machine learning in python

# Task 1: Input optimization.
Open the `src/input_opt.py` file. The network `./data/weights.pkl` contains network weights pre-trained on MNIST. Turn the network optimization problem around, and find an input that makes a particular output neuron extremely happy. In other words maximize,

```math
\max_\mathbf{x} y_i = f(\mathbf{x}, \theta) .
```

Use `jax.value_and_grad` to find the gradients of the network input $\mathbf{x}$.
Start with a `jax.random.uniform` network input of shape `[1, 28, 28, 1]` and 
iteratively optimize it. Execute your script with `python src/input_opt.py`.

# Task 2 Integrated Gradients (Optional):


Reuse your MNIST digit recognition code. Implement IG as discussed in the lecture. Recall the equation

```math
\text{IntegratedGrads}_i(x) = (x_i - x_i') \cdot \sum_{k=1}^m \frac{\partial F (x' + \frac{k}{m} \cdot (x - x'))}{\partial x_i}.
```

F partial xi denotes the gradients with respect to the input color-channels i.
x prime denotes a baseline black image. And x symbolizes an input we are interested in.
Finally, m denotes the number of summation steps from the black baseline image to the interesting input.

Follow the TODOs in `./src/mnist_integrated.py` and then run `scripts/integrated_gradients.slurm`.

# Task 3 - Deepfake detection (Optional):
In this exercise we will consider 128 by 128-pixel fake images from [StyleGAN](https://github.com/NVlabs/stylegan) and pictures of real people from the  [Flickr-Faces-HQ](https://github.com/NVlabs/ffhq-dataset) dataset.

Flickr-Faces-HQ images depict real people, such as the person below:

![real person](./figures/real.png)

Generative adversarial networks allow the generation of fake images at scale. Does the picture below seem real? 

![fake person](./figures/fake.png)

How can we identify the fake? Given that modern neural networks can generate hundreds of fake images per second can we create a classifier to automate the process?

### 3.1 Getting started:
1. Move to the `data` folder in your terminal. Download [ffhq_style_gan.zip](https://drive.google.com/uc?id=1MOHKuEVqURfCKAN9dwp1o2tuR19OTQCF&export=download) on bender using the command
   ```bash
   gdown https://drive.google.com/uc?id=1MOHKuEVqURfCKAN9dwp1o2tuR19OTQCF
   ```
   If `gdown` is not installed, type `pip install gdown` and then try again.
2. Type `export UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE` to make unzipping big archives possible.
3. Extract the image pairs here by executing `unzip ffhq_style_gan.zip` in the terminal.

The desired outcome is to have a folder called `ffhq_style_gan` in the project data-folder.


### 3.2 Analyzing the data
The `load_folder` function from the `util` module loads both real and fake data.
Code to load the data is already present in the `deepfake_interpretation.py` file.

Compute log-scaled frequency domain representations of samples from both sources via

```math
\mathbf{F}_I =  \log_e (| \mathcal{F}_{2d}(\mathbf(I)) | + \epsilon ), \text{ with } \mathbf{I} \in \mathbb{R}^{h,w,c}, \epsilon \approx 0
```

Above $h$, $w$ and $c$ denote image height, width and columns. $log$ denotes the natural logarithm, and bars denote the absolute value. A small epsilon is added for numerical stability.

Use the numpy functions `jnp.log`, `jnp.abs`, `jnp.fft.fft2`. By default, `fft2` transforms the last two axes. The last axis contains the color channels in this case. We are looking to transform the rows and columns.

Plot mean spectra for real and fake images as well as their difference over the entire validation or test sets. For that complete the TODOs in `src/deepfake_interpretation.py` and run the script `scripts/train.slurm`.


## 3.3 Training and interpreting a linear classifier
Train a linear classifier consisting of a single `nn.Dense`-layer on the log-scaled Fourier coefficients using Flax. Plot the result. What do you see?
