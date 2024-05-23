import numpy as np

def roulette_bit_switching(b=[2, 4, 6, 8], t_l=0.2, t_m=0.5):
    """
    Implements the Roulette algorithm for bit-switching.
    
    Parameters:
    b (list): Candidate bit-widths set.
    t_l (float): The HMT of the current layer.
    t_m (float): The average HMT.
    
    Returns:
    int: Selected bit-width.
    """
    n = len(b)
    r = np.random.uniform(0, 1)
    
    if t_l < t_m:
        p = [1/n] * n
    else:
        L1_norm = sum(b)
        p = [bi / L1_norm for bi in b]
    
    s = 0
    i = 0
    while s < r:
        s += p[i]
        i += 1
        
    return b[i - 1]


if __name__ == '__main__':
    # example
    b = [2, 4, 6, 8]  # Candidate bit width
    t_l = 0.2  # HMT of current layer
    t_m = 0.5  # Average HMT

    selected_bit_width = roulette_bit_switching(b, t_l, t_m)
    print(f"Selected bit-width: {selected_bit_width}")
    
    t_l = 0.7  # HMT of current layer
    selected_bit_width = roulette_bit_switching(b, t_l, t_m)
    print(f"Selected bit-width: {selected_bit_width}")
