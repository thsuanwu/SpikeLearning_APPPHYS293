import numpy as np

def run_train(M, net_param, train_param, time_param, dt, utarg):
    """
    Trains a neural network model.

    Parameters:
    M (numpy.ndarray): Initial weight matrix.
    net_param (list): Network parameters [x, r, b].
    train_param (list): Training parameters [stim, lambda, learn_every, nloop, target_type].
    time_param (list): Time parameters [stim_on, stim_off, train_time].
    dt (float): Time step for the simulation.
    utarg (numpy.ndarray): Target trajectories for training.

    Returns:
    numpy.ndarray: Updated weight matrix after training.
    """
    N = M.shape[1]
    w = np.zeros((N, N))
    extinput = np.zeros(N)
    u = np.zeros(N)

    # Unpack network parameters
    x, r, b = net_param[:N], net_param[N:2*N], net_param[2*N]

    # Unpack training parameters
    stim, lambda_, learn_every, nloop, target_type = train_param[:N], train_param[N+1], train_param[N+2], train_param[N+3], train_param[N+4]
    
    # Unpack time parameters
    stim_on, stim_off, train_time = time_param

    # Define RHS of ODEs
    def dtheta(x_var, I_var):
        return 1 - np.cos(x_var) + I_var * (1 + np.cos(x_var))

    def dr(r_var):
        return -b * r_var

    # Set up correlation matrix
    P = {}
    Px = {}
    for ni in range(N):
        ni_preind = M[ni, :] != 0
        P[ni] = np.eye(np.sum(ni_preind)) / lambda_
        Px[ni] = ni_preind

    t = 0.
    for iloop in range(int(nloop)):
        print(f"Loop {iloop + 1}...\n")
        for i in range(int(train_time / dt)):
            # External stimulus
            if stim_on / dt < i <= stim_off / dt:
                extinput[:] = stim
            else:
                extinput[:] = np.zeros(N)

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

            # Update each neuron's spike
            if ind.size > 0:
                r[idx] += b

            # Update x, r, z, t
            x = x + (k1 + 2 * k2 + 2 * k3 + k4) / 6
            r = r + (l1 + 2 * l2 + 2 * l3 + l4) / 6
            u = M @ r
            t += dt

            # Train w
            if stim_off / dt < i <= train_time / dt and i % (learn_every / dt) == 0:
                for ni in range(N):
                    # Update inverse correlation matrix
                    k = P[ni] @ r[Px[ni]]
                    vPv = r[Px[ni]].T @ k
                    den = 1.0 / (1.0 + vPv)
                    P[ni] -= k[:, None] @ (k[None, :] * den)

                    # Update recurrent weights
                    e = M[ni, Px[ni]] @ r[Px[ni]] - utarg[i, ni]
                    dw = -e * k * den
                    M[ni, Px[ni]] += dw

    return M