import numpy as np

def generate_random_number(limit, seed):
    bits = int(np.log2(limit))
    np.random.seed(seed)
    return int(''.join([str(np.random.randint(0, 2)) for _ in range(bits)]), base=2)

def get_co_prime(n):
    if n % 2 == 1: # If Odd, return 2
        return 2
    # If even return 1
    return 1

def get_lcg_parameters(m):
    m_factors = get_factors(m)
    m_prime_factors = get_prime_numbers(m_factors)
    
    num_prod = 1
    for num in m_prime_factors:
        num_prod *= num
    if m % 4 == 0:
        num_prod *= 4
        
    a = num_prod + 1
    c = get_co_prime(m)
    
    return a, c

def lcg(number_of_samples, m, seed):
    a, c = get_lcg_parameters(m)
    random_numbers = np.zeros(number_of_samples)
    for i in range(number_of_samples):
        seed = (a*seed + c) % m
        random_numbers[i] = seed
    return random_numbers
     
def generate_random_uniform_samples(number_of_samples, seed):
    random_numbers = lcg(number_of_samples, 1000, seed)
    uniform_samples = [num/(1001) for num in random_numbers]
    return uniform_samples

def get_factors(num):
    factors = []
    for i in range(2, num-1):
        if num % i == 0:
            factors.extend([i, int(num/i)])
    return list(set(factors))

def is_prime(num):
    for i in range(2, int(num/2)+1):
        if num % i == 0:
            return False
    return True

def get_prime_numbers(nums_list):
    prime_list = []
    for num in nums_list:
        if (num == 2) or (num % 2 != 0 and is_prime(num)):
            prime_list.append(num)
    return prime_list


if __name__ == '__main__':
    print(generate_random_uniform_samples(5, seed=1000))