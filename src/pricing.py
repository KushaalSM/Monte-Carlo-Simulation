import numpy as np

from processes import get_stochastic_path

def price_asian_option(average_type, iterations, S0, option_type, strike, T, steps, process, process_parameters={}, discretization='euler', seed=1331):
    """
    Payoff is dependent on the mean price of the underlying over the life of the option.
    """
    if process_parameters == {}:
        raise Exception("Process parameters empty")
    if option_type not in ['C', 'P']:
        raise Exception("Invalid Option type")
    if average_type not in ['arithmetic', 'geometric']:
        raise Exception("Invalid Average type")
    
    np.random.seed(seed)

    option_prices = np.zeros(iterations)
    r = process_parameters['r']
    option_multiplier = 1 if option_type == 'C' else -1
    for i in range(iterations):
        # Get the path of the underlying
        S = get_stochastic_path(S0, process, process_parameters, T, steps, discretization)
        # Calculate the average price.
        S_avg = np.mean(S) if average_type == 'arithmetic' else np.prod(S) ** (1/steps)
        payoff = max(option_multiplier * (S_avg - strike), 0)
        option_prices[i] = np.exp(-r*T) * payoff

    option_price = np.mean(option_prices)

    print(f"Mean of Option Price : {option_price}")
    print(f"Standard Deviation of Option Price : {np.std(option_prices)}")
    
    return option_price

def price_single_barrier_option(barrier_value, barrier_type, rebate, iterations, S0, option_type, strike, T, steps, process, process_parameters={}, discretization='euler', seed=1331):
    """
    
    """
    if process_parameters == {}:
        raise Exception("Process parameters empty")
    if option_type not in ['C', 'P']:
        raise Exception("Invalid Option type")
    if barrier_type not in ['up-in', 'up-out', 'down-in', 'down-out']:
        raise Exception("Invalid Barrier type")

    np.random.seed(seed)

    option_prices = np.zeros(iterations)
    r = process_parameters['r']
    option_multiplier = 1 if option_type == 'C' else -1
    dt = T / steps
    for i in range(iterations):
        # Get the path of the underlying
        S = get_stochastic_path(S0, process, process_parameters, T, steps, discretization)

        # Calculate the payoff at expiry based on the type of barrier option.
        if barrier_type == 'up-in':
            payoff = max(option_multiplier * (S[-1] - strike), 0) if max(S) >= barrier_value else rebate
        elif barrier_type == 'down-in':
            payoff = max(option_multiplier * (S[-1] - strike), 0) if min(S) <= barrier_value else rebate
        elif barrier_type == 'up-out':
            payoff = max(option_multiplier * (S[-1] - strike), 0) if max(S) <= barrier_value else rebate * \
                np.exp(r * dt * (steps - [i + 1 for i, price in enumerate(S) if price >= barrier_value][0])) # Rebate
        else:
            payoff = max(option_multiplier * (S[-1] - strike), 0) if min(S) >= barrier_value else rebate * \
                np.exp(r * dt * (steps - [i + 1 for i, price in enumerate(S) if price >= barrier_value][0])) # Rebate
        option_prices[i] = np.exp(-r*T) * payoff
    
    option_price = np.mean(option_prices)

    print(f"Mean of Option Price : {option_price}")
    print(f"Standard Deviation of Option Price : {np.std(option_prices)}")
    
    return option_price

