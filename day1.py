import numpy as np

def count_increment(sonar_meas : np.ndarray) -> float:
    depth_diff = np.diff(sonar_meas)
    return np.sum(depth_diff > 0)

input_depth = np.loadtxt("input1")
print(f"Ans: {count_increment(input_depth)}")

filter_size = 3
filter = np.ones(filter_size)
rolling = np.convolve(input_depth, filter)[filter_size-1:-2]
print(f"Part2 ans: {count_increment(rolling)}")