"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np

def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """

    f = gzip.open('data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding = 'latin1')
    f.close()
    return (training_data, validation_data, test_data)

def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""

    """
    I am adding my own comments here. It seems like all we want to do, is just to reshape the vectors, such that they
    are interpreted as column vectors by numpy. That is all.
    """
    tr_d, va_d, te_d = load_data()

    # Training inputs is orginally a list of 784D vectors.
    # Training inputs is reshaped to be a list of column vectors, each column vector represents one picture.
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    
    # Training results is a list which has 50_000 entries. Each entry is the number which the corresponding picture should have as target.
    # We are using this number to make a 10D-vector which should be the target for our neural network.
    training_results = [vectorized_result(y) for y in tr_d[1]]
    
    # Finishing up the training data.
    training_data = list(zip(training_inputs, training_results))

    # Validation inputs is also a list 2-tuple, where the first entry is a list of 10_000 784-dimensional arrays
    # We are reshaping each picture, such that it becomes a column vector
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]

    # Validation data is of the form ([columnVector1, columnVector 2, ...], [target1, target2, ...])
    # For validation data and test data, the targets are simply integers representing the number on the picture.
    validation_data = list(zip(validation_inputs, va_d[1]))
    
    # Doing the same procedure on the test data as we did on the validation data.
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    
    # With my own words, i would say that this creates a column vector which represents the target for our network.
    # Our network has 10 outputs, so the target for the network must be a 10D column vector.
    # We set the target of the network to be 0's for all numbers except the correct number, which should be labelled as 1.
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e


    
