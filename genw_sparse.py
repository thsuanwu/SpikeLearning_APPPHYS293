# Importing necessary libraries
import numpy as np
from scipy.stats import norm, bernoulli

def genw_sparse(N, m, sd, p):
    """
    Generates a sparse weight matrix for a neural network.

    Parameters:
    N (int): Number of neurons.
    m (float): Mean of the normal distribution for the weights.
    sd (float): Standard deviation of the normal distribution for the weights.
    p (float): Probability of having a connection between two neurons.

    Returns:
    numpy.ndarray: A generated sparse weight matrix of size N x N.
    """

    # Ensure N is an integer in case a float is passed
    N = int(N)

    # Initialize an N x N matrix with zeros
    w = np.zeros((N, N))

    # Recurrent connections
    for i in range(N):
        for j in range(N):
            # Generate a random number to decide if there should be a connection
            # based on the probability p (Bernoulli distribution)
            if np.random.rand() < p:
                # If a connection is to be made, assign a weight based on a normal
                # distribution with mean 'm' and standard deviation 'sd'
                w[i, j] = norm.rvs(m, sd)

    # No autapse (neurons do not connect to themselves)
    # Set the diagonal elements of the matrix to 0
    np.fill_diagonal(w, 0)

    # Return the generated weight matrix
    return w