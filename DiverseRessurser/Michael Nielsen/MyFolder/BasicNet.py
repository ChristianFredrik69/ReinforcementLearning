import numpy as np
import random
import mnist_loader


class Network:

    # np.randn gives you an array with the specified size, where each entry is a number around 0, there is some standard distribution.
    # In any case, the way the weights and biases are set up now, makes it so that biases[0] gives you the biases of the first layer,
    # weights[0] gives you the weights of the first layer, and so forth.
    # Essentially, we are storing the weights and biases in 2D matrices.

    def __init__(self, sizes):
        """
        Sizes gives the number of neurons in each layer.
        [2, 3, 1] will give you an input layer of size 2,
        hidden layer of size 3, and output layer of size 1.
        """
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    
    def feedforward(self, a):
        """
        Returns the outputof the network, provided that a is the input.
        It is important to note that this method expects a (n, 1) array is input, not a (n,) vector.
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a
    
    def SGD(self, training_data, epochs, mini_batch_size, alfa, test_data = None):
        """
        Training the neural network with the stochastic gradient descent algorithm.
        Training data is a list of tuples (x, y) representing what the input is and what the desired output is.
        """
        
        n = len(training_data)
        
        if test_data:
            n_test = len(test_data)
        
        for i in range(epochs):
            
            # Creating mini-batches and calling the update-method.
            np.random.shuffle(training_data)

            # Here we are creating mini-batches by slicing the entire batch of 50_000 images or so.
            mini_batches = [training_data[k : k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            # For each mini-batch, we calculate the gradient and perform a small update to the weights
            # with the use of the stochastic gradient descent algorithm.
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, alfa)

            # This is just printing the performance of the neural net after each epoch.
            if test_data:
                print(f'Epoch {i + 1}: {self.evaluate(test_data)} / {n_test}')
            else:
                print(f'Epoch {i + 1} complete')
    
    def update_mini_batch(self, mini_batch, eta):
        """
        This is the part where we are actually updating the network's weights and biases with the
        stochastic gradient descent algorithm.
        """
        
        # Setting up arrays which have the same shape as the matrices for the weights and biases.
        # For the biases, we have a list of column vectors. It is perhaps not correct to call these things matrices,
        # as the shape of something like nabla_b is [(30, 1), (10, 1)] if you have a net with shape [784, 30, 10].
        # In practice, this means that nabla_b is a list with column vectors.
        nabla_b = [np.zeros(b.shape) for b in self.biases]

        # Nabla_w will be a list of matrices. If you have a net like [784, 30, 10], then the first matrix will have a shape of (30, 784).
        # Essentially, matrix number 1 will have shape (layer2_num_neurons, layer1_num_neurons), with indexing starting at 1.
        # So this means that entry number x will have the shape (layerX_num_neurons, layerX-1_num_neurons).
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # In this part we are just adding up the gradients for each training example.
        # Backpropagation is doing most of the work here.
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # We have the sum of gradient changes that we want for each training example. 
        # To find the step-size in the direction of the gradient, we divide the sum og gradients by number of samples,
        # and then we multiply the result by the learning rate.
        # Since we are doing gradient DESCENT, we are taking steps in the NEGATIVE direction of the gradient.
        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
    
    def backprop(self, x, y):

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = [] # List to store all the z-vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        # Backward pass
        delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """
        Returns the number of correct results.
        The output is assumed to be the argmax of output layer.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) for x, y in test_data]
        return sum( x == y for x, y in test_results)
    
    def cost_derivative(self, output_activations, y):
        """
        Returns a vector of partial derivatives for the output activations?
        This is used in backpropagation it seems.
        Also there is this comment from the chatbot saying that the derivative
        depends on the cost function you have, and that the derivative for MSE
        just happens to be particularly simple.
        """
        return (output_activations - y)

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

if __name__ == '__main__':

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data = test_data)
    # net = Network([784, 10])
    # net.SGD(training_data, 30, 50, 3, test_data = test_data)





