# Exercise: Interpretable machine learning in python

# Task 1: Input optimization.
Open the `src/input_opt.py` file. The network `./data/weights`.pkl` contains network weights pre-trained on MNIST. Turn the network optimization problem around, and find an input that makes a particular output neuron extremely happy. In other words maximize,

$$ \max_\mathbf{x} y_i = f(\mathbf{x}, \theta) .$$

Use `jax.value_and_grad` to find the gradients of the network input $\mathbf{x}$.
Start with a `jax.random.uniform` network input of shape `[1, 28, 28, 1]` and 
iteratively optimize it.

# Task 2: Integrated Gradients (Optional):


Reuse your MNIST digit recognition code. Implement IG as discussed in the lecture. Recall the equation

$$ \text{IntegratedGrads}_i(x) = (x_i - x_i') \cdot \sum_{k=1}^m \frac{\partial F (x' + \frac{k}{m} \cdot (x - x'))}{\partial x_i}. $$

F partial xi denotes the gradients with respect to the input color-channels i.
x prime denotes a baseline black image. And x symbolizes an input we are interested in.
Finally, m denotes the number of summation steps from the black baseline image to the interesting input.

Follow the todos in `./src/mnist_integrated`.


