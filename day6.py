import numpy as np

def parse_input(fname : str) -> np.ndarray:
    '''
    Parse input state
    '''
    initial_state = np.loadtxt(fname, delimiter=",")
    return initial_state

def step(fish_state : np.ndarray) -> np.ndarray:
    '''
    '''
    # 1. Check if fish reaches zero state
    mask = fish_state == 0

    # 2. Reset zeros to 7, add num of zeros to end with 9
    num_birth = np.sum(mask)
    fish_state[mask] = 7
    fish_state = np.hstack((fish_state, 9 * np.ones(num_birth)))

    # 3. Minus day
    fish_state -= 1
    return fish_state

def particle_methods(init_state : np.ndarray, num_days : int) -> int:
    '''
    Naive particle ways of getting answer
    '''
    state = init_state.copy().astype(int)
    for i in range(num_days):
        state = step(state)

    return len(state)

def binning_method(initial_state : np.ndarray, num_days : int) -> int:
    '''
    Count number of fish in each life cycle
    '''
    
    fish_bins = np.bincount(initial_state.astype(int), minlength=9)
    for i in range(num_days):
        fish_bins = np.roll(fish_bins, -1)
        fish_bins[6] += fish_bins[-1]
    return np.sum(fish_bins)

fname = "input6"
state = parse_input(fname)

part1_days, part2_days = 80, 256
part1_ans = particle_methods(state.copy(), part1_days)
print(f"{part1_days} days of fish: {part1_ans}")

part2_ans = binning_method(state.copy(), part2_days)
print(f"{part2_days} days of fish: {part2_ans}")
