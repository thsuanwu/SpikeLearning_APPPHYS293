import numpy as np

def run_test(M, net_param, time_param, dt):
    """
    Runs a test simulation on the neural network model.

    Parameters:
    M (numpy.ndarray): Weight matrix of the neural network.
    net_param (list): Network parameters [x, r, b, stim].
    time_param (list): Time parameters [stim_on, stim_off, test_time].
    dt (float): Time step for the simulation.

    Returns:
    numpy.ndarray: The test output of the network over time.
    """
    N = M.shape[1]
    extinput = np.zeros(N)
    u = np.zeros(N)

    # Unpack network parameters
    x, r, b, stim = net_param[:N], net_param[N:2*N], net_param[2*N], net_param[2*N+1:3*N+1]
    
    # Unpack time parameters
    stim_on, stim_off, test_time = time_param

    # Define RHS of ODEs
    def dtheta(x_var, I_var):
        return 1. - np.cos(x_var) + I_var * (1. + np.cos(x_var))

    def dr(r_var):
        return -b * r_var

    t = 0.

    # Initialize array to store test data
    utest = np.zeros((int(test_time / dt), N))

    for i in range(int(test_time / dt)):
        if stim_on / dt < i <= stim_off / dt:
            extinput[:] = stim
        else:
            extinput[:] = np.zeros(N)

        # Define synaptic activity
        u = M @ r

        # Update neuron phase using RK4 integration
        k1 = dt * dtheta(x, u + extinput)
        k2 = dt * dtheta(x + k1 / 2, u + extinput)
        k3 = dt * dtheta(x + k2 / 2, u + extinput)
        k4 = dt * dtheta(x + k3, u + extinput)
        xnext = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Update filtered spikes
        l1 = dt * dr(r)
        l2 = dt * dr(r + l1 / 2)
        l3 = dt * dr(r + l2 / 2)
        l4 = dt * dr(r + l3)

        # Spike detection
        idx = (xnext - x > 0) & (xnext - x > np.mod(np.pi - np.mod(x, 2 * np.pi), 2 * np.pi))
        ind = np.where(idx)[0]

        # Update each neuron's spikes
        if ind.size > 0:
            r[idx] += b

        # Update x, r
        x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
        r = r + (l1 + 2 * l2 + 2 * l3 + l4) / 6
        u = M @ r
        t += dt

        # Save test data
        utest[i, :] = u

    return utest