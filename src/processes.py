import numpy as np

from sampling import get_gaussian_samples

class GBM:
    def __init__(self, mu, sigma, param_type='constant') -> None:
        """
        mu : drift of the process
        sigma : volatility
        """
        self.mu = mu
        self.sigma = sigma
        self.param_type = param_type
        
        # Verify the correctness of parameter types.
        self.check_param_type()

    def check_param_type(self):
        """
        Checks if the parameters passed the and param_type match.
        """
        if self.param_type == 'constant' and (not isinstance(self.mu, float) or not isinstance(self.sigma, float)):
            raise Exception(f"Parameters must be constants for the param_type '{self.param_type}'")
        if self.param_type == 'term_structure' and (not callable(self.mu) or not callable(self.sigma)):
            raise Exception(f"Parameters must be callable functions for the param_type '{self.param_type}'")
        return

    def generate_path(self, X0, T, steps, seed=1331):
        dt = T / steps
        X = np.zeros(steps)
        X[0] = X0
        for i in range(1, steps):
            if self.param_type == 'constant':
                mu = self.mu
                sigma = self.sigma
            elif self.param_type == 'term_structure':
                mu = self.mu(i*dt)
                sigma = self.sigma(i*dt)
            X[i] = X[i-1] * np.exp((mu - 0.5 * sigma**2) * dt + \
                sigma * np.sqrt(dt) * get_gaussian_samples(0, 1, 1, seed))
        return X

class OU:
    def __init__(self, mu, thetha, sigma) -> None:
        self.mu = mu
        self.thetha = thetha
        self.sigma = sigma

    def generate_path(self, X0, T, steps, discretization='euler', z_array=[], seed=1331):
        if not discretization in ['euler', 'millstein', 'runge-kutta']:
            raise Exception(f"Invalid discretization form {discretization}")

        if discretization in ['euler', 'millstein', 'runge-kutta']:
            return self.generate_euler_path(X0, T, steps, z_array, seed)

    def generate_euler_path(self, X0, T, steps, z_array=[], seed=1331):
        if not z_array == [] and len(z_array) != steps:
            raise Exception("Not sufficient Gaussian samples")
        dt = T / steps
        X = np.zeros(steps)
        X[0] = X0
        for i in range(1, steps):
            X[i] = X[i-1] + self.thetha * (self.mu - X[i-1]) *  dt + self.sigma * np.sqrt(dt) * (get_gaussian_samples(0, 1, 1, seed) if z_array == [] else z_array[i])
        return X