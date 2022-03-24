import numpy as np

from sampling import get_gaussian_samples

def get_discretized_OU_process(X0, mu, thetha, sigma, T, steps, z_array=[], seed=1331):
    if not z_array == [] and len(z_array) != steps:
        raise Exception("Not sufficient Gaussian samples")
    dt = T / steps
    X = np.zeros(steps)
    X[0] = X0
    for i in range(1, steps):
        X[i] = X[i-1] + thetha * (mu - X[i-1]) *  dt + sigma * np.sqrt(dt) * (get_gaussian_samples(0, 1, 1, seed) if z_array == [] else z_array[i])
    return X

def get_discretized_CIR_process(X0, mu, thetha, sigma, T, steps, z_array=[], seed=1331):
    if not z_array == [] and len(z_array) != steps:
        raise Exception("Not sufficient Gaussian samples")
    dt = T / steps
    X = np.zeros(steps)
    X[0] = X0
    for i in range(1, steps):
        X[i] = X[i-1] + thetha * (mu - X[i-1]) *  dt + sigma * np.sqrt(dt * X[i-1]) * (get_gaussian_samples(0, 1, 1, seed) if z_array == [] else z_array[i])
    return X

def get_discretized_Heston_process(X0, v0, rho, r, q, mu, sigma, thetha, T, steps, seed=1331):
    dt = T / steps
    X = np.zeros(steps)
    z1 = np.random.normal(0, 1, steps)
    z = np.random.normal(0, 1, steps)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * z
    v = get_discretized_CIR_process(v0, mu, thetha, sigma, T, steps, z2, seed)
    X[0] = X0
    for i in range(1, steps):
        X[i] = X[i-1] + (r-q) * X[i-1] * dt + np.sqrt(v[i-1]) * X[i-1] * np.sqrt(dt) * z1[i]
    return X


def get_discretized_Variance_Gamma_process(X0, r, q, mu, sigma, nu, h, steps, seed=1331):
    np.random.seed(seed)
    X = np.zeros(steps)
    X[0] = X0
    w = (1 / nu) * np.log(1 - mu*nu - 0.5*nu*sigma**2)
    for i in range(1, steps):
        z = get_gaussian_samples(0, 1, 1, seed)
        g = np.random.gamma(h/nu, nu, 1)
        x = mu * g + sigma * np.sqrt(g) * z
        X[i] = X[i-1] * np.exp((r-q) * h + w * h + x)
    return X

def get_discretized_Merton_Jump_process(X0, r, q, sigma, dt, lambda_, kappa, eta, delta, steps, seed=1331):
    np.random.seed(seed)
    X = np.zeros(steps)
    X[0] = X0
    
    for i in range(1, steps):
        n_jumps = np.random.poisson(lambda_*dt, 1)
        J = 0
        for _ in range(1, n_jumps):
            J += eta + delta * get_gaussian_samples(0, 1, 1, seed)
        
        z = get_gaussian_samples(0, 1, 1, seed)
        X[i] = X[i-1] * np.exp((r - q - (lambda_ * kappa) - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z + J)
    
    return X




