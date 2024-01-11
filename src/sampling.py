from json.tool import main
import numpy as np

def get_uniform_samples(lower, upper, number_of_samples):
    if number_of_samples == 1:
        return np.random.uniform(lower, upper)
    return np.random.uniform(lower, upper, size=number_of_samples)

def get_gaussian_samples(mean, stdev, number_of_samples):
    if number_of_samples == 1:
        return np.random.normal(mean, stdev)
    return np.random.normal(mean, stdev, size=number_of_samples)

def get_2D_quasi_uniform_samples(low, high, number_of_samples, number_of_blocks):
    block_size = (high - low) / np.sqrt(number_of_blocks)
    idx = 0
    samples = []
    h = low
    v = low
    while idx < number_of_samples:
        u1 = get_uniform_samples(0, 1, 1)
        u2 = get_uniform_samples(0, 1, 1)

        samples.append([h + block_size * u1, v + block_size * u2])
        h += block_size
        if h >= high:
            h = low
            v += block_size
            if v >= high:
                v = low
        idx += 1
    return samples


if __name__ == '__main__':
    np.random.seed(1331)
    print(get_uniform_samples(0, 1, 5))



