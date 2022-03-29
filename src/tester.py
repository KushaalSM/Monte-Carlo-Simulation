import numpy as np


def run_sampler(seed):
    np.random.seed(seed)
    return [gen_samp(), gen_samp(), gen_samp(), gen_samp(), gen_samp(), gen_samp()]

def gen_samp():
    return np.random.normal(0, 1)

if __name__ == '__main__':
    print(run_sampler(seed=1331))