import numpy as np
import tempfile

# Read File input
file_name = "input4"
lines = None
with open(file_name, 'r') as f:
    lines = f.readlines()
commands = lines[0].strip()
commands = commands.split(",")

# Read Matrix through tempFile so that loadtxt works
try:
    matricies = lines[1:]
    f = tempfile.NamedTemporaryFile(mode='w+t')
    f.writelines(matricies)
    f.seek(0)
    mat = np.loadtxt(f.name)
finally:
    f.close()

def calculate_score(mat : np.ndarray, mask : np.ndarray, command : float) -> float:
    '''
    Calculate Score with matrix and mask
    '''
    return command * np.sum(mat[np.logical_not(mask)])

# Reshape Matrix to N x 5 x 5 Matrices
(_, n) = mat.shape
n_dim_mat = mat.reshape((-1, n, n))
mask = np.zeros_like(n_dim_mat, dtype=int)

score = -1
term_mask = None

once = True
for num in commands[:-1]:
    # Update Mask
    this_num = int(num)
    mask[n_dim_mat == this_num] = 1

    # Check Termination Condition, since diagonal doesn't count
    hori_sum = np.sum(mask, axis=2)
    vert_sum = np.sum(mask, axis=1)

    if np.any(np.isclose(hori_sum, 5)):
        term_mask = hori_sum
    elif np.any(np.isclose(vert_sum, 5)):
        term_mask = vert_sum

    if term_mask is not None:
        indices = np.argwhere(np.isclose(term_mask, n))
        score = calculate_score(n_dim_mat[indices[0, 0]], mask[indices[0, 0]], command = this_num)
        if once:
            print(f"Part 1 Answer: {score}")
            once = False

        n_dim_mat[indices[:, 0]] = -1
        mask[indices[:, 0]] = -1
        term_mask = None

print(f"Part 2 Answer: {score}")
