import numpy as np

from sampling import get_gaussian_samples

def generate_brownian_motion_path(T, number_of_samples, seed=1331):
    dt = T / number_of_samples
    B = np.zeros(number_of_samples)
    for i in range(1, number_of_samples):
        B[i] = B[i-1] + np.sqrt(dt) * get_gaussian_samples(mean=0, stdev=1, number_of_samples=1, seed=seed)
    return B


def generate_geometric_brownian_motion_path(mu, sigma, initial_point, T, number_of_samples, seed):
    dt = T / number_of_samples
    S = np.zeros(number_of_samples)
    S[0] = initial_point
    for i in range(1, number_of_samples):
        z = get_gaussian_samples(mean=0, stdev=1, number_of_samples=1, seed=seed)
        S[i] = S[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z )
    return S

def generate_brownian_bridge_path(start, end, t0, T, number_of_samples, seed=1331):
    dt = (T - t0) / number_of_samples
    BB = np.zeros(number_of_samples + 2)
    BB[0] = start
    BB[-1] = end
    t = t0
    for i in range(1, number_of_samples + 1):
        t += dt
        z = get_gaussian_samples(mean=0, stdev=1, number_of_samples=1, seed=seed)
        BB[i] = BB[i-1] + (end - BB[i-1]) * (dt / (T - t - dt)) + np.sqrt((T - t) * dt / (T - t- dt)) * z
    return BB



