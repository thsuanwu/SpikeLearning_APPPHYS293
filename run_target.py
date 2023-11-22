import numpy as np
from scipy.stats import norm

def run_target(M, net_param, target_type):
    """
    Generates target trajectories based on the specified target type.

    Parameters:
    M (numpy.ndarray): The connectivity matrix.
    net_param (list): List containing network parameters [N, p, g, b, dt, T].
    target_type (str): Type of target to generate ('periodic', 'ratemodel', or 'ou').

    Returns:
    numpy.ndarray: Generated target trajectories.
    """

    # Unpacking network parameters
    N, p, g, b, dt, T = net_param
    N = int(N)

    # Create a time vector
    t = np.arange(dt, T + dt, dt)

    # 1. Periodic functions
    if target_type == "periodic":
        u_periodic = np.zeros((len(t), N))
        for j in range(N):
            A = 0.5 + np.random.rand()
            T1 = 50. + 50. * np.random.rand()
            T2 = 10. + 40. * np.random.rand()
            t1 = T1 * np.random.rand()
            t2 = T2 * np.random.rand()

            # Generate periodic target patterns
            u_periodic[:, j] = A * np.sin((t - t1) * (2 * np.pi / T1)) * np.sin((t - t2) * (2 * np.pi / T2))

        return u_periodic

    # 2. Rate network model
    if target_type == "ratemodel":
        u_rm = np.zeros((len(t), N))
        g_rm = 5
        M = g_rm * genw_sparse(N, 0, 1, p) / np.sqrt(N * p)
        b_rm = 1 / 4

        # Adjust the matrix M so the mean of incoming synaptic connections is zero
        for i in range(N):
            idx = np.abs(M[i, :]) > 0
            M[i, idx] -= np.mean(M[i, idx])

        # Random initial condition
        u_rm[0, :] = 0.2 * np.random.rand(N)

        # Non-linear function phi
        def phi(x):
            x[x < 0] = 0
            return np.sqrt(x) / np.pi

        # Generate rate model target patterns
        for i in range(len(t) - 1):
            u_rm[i + 1, :] = (1 - dt * b_rm) * u_rm[i, :] + dt * b_rm * M.dot(phi(u_rm[i, :]))

        return u_rm

    # 3. Ornstein-Uhlenbeck process
    if target_type == "ou":
        u_ou = np.zeros((len(t), N))
        b_ou = 1 / 20
        mu = 0.0
        sig = 0.3

        # Generate OU target patterns
        for j in range(N):
            for i in range(len(t) - 1):
                u_ou[i + 1, j] = u_ou[i, j] + b_ou * (mu - u_ou[i, j]) * dt + sig * np.sqrt(dt) * norm.rvs()

        return u_ou

# Note: Remember to import or define the `genw_sparse` function as used in the 'ratemodel' section.
