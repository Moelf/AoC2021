# Too much of a havoc on the prompt side, not going to document

import numpy as np

fname = "input3"
lines = None
with open(fname, 'r') as f:
    lines = f.readlines()

col = len(lines[0].strip())
row = len(lines)

data_mat = np.zeros((row, col))
for i, l in enumerate(lines):
    l = l.strip()
    for j in range(col):
        data_mat[i, j] = int(l[j])

gamma = np.round(np.sum(data_mat, axis=0) / row).astype(int)
epsilon = 1 - gamma

def matrix_bit_to_num(bit_mat : np.ndarray) -> float:
    power = np.arange(len(bit_mat))[::-1]

    gamma_val = np.sum(np.power(bit_mat * 2, power))

    if bit_mat[-1] == 0:
        gamma_val -= 1
    
    return gamma_val

print(f'prod: {matrix_bit_to_num(gamma) * matrix_bit_to_num(epsilon)}')

oxygen_mask = np.ones(row, dtype=bool)
oxygen_string = np.zeros(col)
for i in range(col):
    this_mat = data_mat[oxygen_mask, :]

    this_rating = int((np.sum(this_mat[:, i]) * 2) >= len(this_mat))
    oxygen_string[i] = this_rating
    
    this_mask = this_mat[:, i] == this_rating
    oxygen_mask[oxygen_mask] = np.bitwise_and(oxygen_mask[oxygen_mask], this_mask)

scrubber_mask = np.ones(row, dtype=bool)
scrubber_string = np.zeros(col)
for i in range(col):
    this_mat = data_mat[scrubber_mask, :]
    
    if len(this_mat) == 1:
        scrubber_string[i:] = this_mat[0, i:]
        break
    this_rating = int((np.sum(this_mat[:, i]) * 2) < len(this_mat))
    scrubber_string[i] = this_rating
    
    this_mask = this_mat[:, i] == this_rating
    scrubber_mask[scrubber_mask] = np.bitwise_and(scrubber_mask[scrubber_mask], this_mask)

print(f"prod: {matrix_bit_to_num(oxygen_string) * matrix_bit_to_num(scrubber_string)}")
