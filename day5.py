import numpy as np

def parse_input(fname : str) -> np.ndarray:
    '''
        Parse Inputs
    '''
    lines = None
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    ret_map = np.zeros((len(lines), 4), dtype=int)
    for idx, l in enumerate(lines):
        l = l.strip()
        start, end = l.split(" -> ")
        start = start.split(",")
        end = end.split(",")
        ret_map[idx, [0, 1]] = int(start[0]), int(start[1])
        ret_map[idx, [2, 3]] = int(end[0]), int(end[1])

    return ret_map

def create_map_from_input(input : np.ndarray) -> np.ndarray:
    '''
    Create Map from input
    '''

    max_x = np.max(input[:, [0, 2]]).astype(int)
    max_y = np.max(input[:, [1, 3]]).astype(int)
    print(f"max_x: {max_x}")
    print(f"max_y: {max_y}")
    return np.zeros((max_y + 1, max_x + 1))

def map_world(map : np.ndarray, input: np.ndarray, part1 : bool = True) -> np.ndarray:
    '''
    Generate Map
    '''
    for obs in input:
        min_x = np.min(obs[[1,3]])
        max_x = np.max(obs[[1,3]])
        min_y = np.min(obs[[0,2]])
        max_y = np.max(obs[[0,2]])

        if (min_x != max_x) and (min_y != max_y):
            # Non axial
            if part1:
                continue
            x_idx = np.arange(min_x, max_x + 1)
            y_idx = np.arange(min_y, max_y + 1)
            if obs[0] != min_y:
                y_idx = y_idx[::-1]
            
            if obs[1] != min_x:
                x_idx = x_idx[::-1]

            map[x_idx, y_idx] += 1

        else:
            map[min_x : max_x + 1, min_y:max_y + 1] += 1
    return map

def count_overlap(map : np.ndarray, num_overlap : int) -> int:
    '''
    Number of cells with 2 or larger
    '''
    return np.sum(map >= num_overlap)

fname = "input5"
observation = parse_input(fname)
empty_map = create_map_from_input(observation)
filled_map = map_world(empty_map.copy(), observation)
print(f"Part1: {count_overlap(filled_map, 2)}")

part2_filled_map = map_world(empty_map.copy(), observation, part1=False)
print(f"Part2: {count_overlap(part2_filled_map, 2)}")

