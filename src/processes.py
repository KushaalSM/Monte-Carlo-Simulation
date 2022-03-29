import numpy as np

from sampling import get_gaussian_samples

def get_stochastic_path(X0, process, process_parameters, T, steps, discretization='euler'):
    if process == 'GBM':
        gbm = GBM(process_parameters['mu'], process_parameters['sigma'])
        return gbm.generate_path(X0, T, steps)
    
    if process == 'HESTON':
        heston = Heston(process_parameters['r'], process_parameters['q'], process_parameters['mu'], process_parameters['thetha'], \
            process_parameters['sigma'], process_parameters['rho'])
        return heston.generate_path(X0, process_parameters['V0'], T, steps, discretization, z_array=[])

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

    def generate_path(self, X0, T, steps):
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
                sigma * np.sqrt(dt) * get_gaussian_samples(0, 1, 1))
        return X

class OU:
    def __init__(self, mu, thetha, sigma) -> None:
        self.mu = mu
        self.thetha = thetha
        self.sigma = sigma

    def generate_path(self, X0, T, steps, discretization='euler', z_array=[]):
        if not discretization in ['euler', 'millstein', 'runge-kutta']:
            raise Exception(f"Invalid discretization form {discretization}")

        if discretization in ['euler', 'millstein', 'runge-kutta']:
            return self.generate_euler_path(X0, T, steps, z_array)

    def generate_euler_path(self, X0, T, steps, z_array=[]):
        if not z_array == [] and len(z_array) < steps:
            raise Exception("Not sufficient Gaussian samples")
        dt = T / steps
        X = np.zeros(steps)
        X[0] = X0
        if z_array == []:
            z_array = get_gaussian_samples(0, 1, steps-1)
        for i in range(1, steps):
            X[i] = X[i-1] + self.thetha * (self.mu - X[i-1]) *  dt + self.sigma * np.sqrt(dt) * z_array[i-1]
        return X

class CIR:
    def __init__(self, mu, thetha, sigma, func_list=[]) -> None:
        self.mu = mu
        self.thetha = thetha
        self.sigma = sigma
        self.func_list = func_list if not func_list == [] else [lambda x: x, lambda x: x, lambda x: x]

    def generate_path(self, X0, T, steps, discretization='euler', z_array=[]):
        if not discretization in ['euler', 'millstein', 'runge-kutta']:
            raise Exception(f"Invalid discretization form {discretization}")

        if discretization in ['euler', 'millstein', 'runge-kutta']:
            return self.generate_euler_path(X0, T, steps, z_array)
    
    def generate_euler_path(self, X0, T, steps, z_array=[]):
        if not z_array == [] and len(z_array) < steps:
            raise Exception("Not sufficient Gaussian samples")
        dt = T / steps
        X = np.zeros(steps)
        X[0] = X0
        if z_array == []:
            z_array = get_gaussian_samples(0, 1, steps-1)

        for i in range(1, steps):
            X[i] = self.func_list[0](X[i-1]) + \
                self.thetha * (self.mu - self.func_list[1](X[i-1])) * dt + \
                    self.sigma * np.sqrt(dt * self.func_list[2](X[i-1])) * z_array[i-1]
        
        return X

class Heston:
    def __init__(self, r, q, mu, thetha, sigma, rho) -> None:
        self.r = r
        self.q = q
        self.mu = mu
        self.thetha = thetha
        self.sigma = sigma
        self.rho = rho

    def generate_path(self, X0, V0, T, steps, discretization='euler', z_array=[]):
        if not discretization in ['euler', 'millstein', 'runge-kutta']:
            raise Exception(f"Invalid discretization form {discretization}")

        if discretization in ['euler', 'millstein', 'runge-kutta']:
            return self.generate_euler_path(X0, V0, T, steps, z_array)

    def generate_euler_path(self, X0, V0, T, steps, z_array=[]):
        if not z_array == [] and len(z_array) < steps:
            raise Exception("Not sufficient Gaussian samples")
        
        dt = T / steps
        X = np.zeros(steps)
        
        if z_array == []:
            z_array = get_gaussian_samples(0, 1, steps-1)
        
        cir_process = CIR(self.mu, self.thetha, self.sigma, func_list=[lambda x: max(x, 0), lambda x: max(x, 0), lambda x: max(x, 0)])
        V = cir_process.generate_path(V0, T, steps, 'euler', z_array)

        z_array = self.rho * z_array + np.sqrt(1 - self.rho**2) * get_gaussian_samples(0, 1, steps-1)

        X[0] = X0
        for i in range(1, steps):
            X[i] = X[i-1] + \
                (self.r - self.q) * X[i-1] * dt + \
                    X[i-1] * np.sqrt(max(V[i], 0)) * np.sqrt(dt) * z_array[i-1]
        
        return X
    