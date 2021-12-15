from collections import defaultdict
import numpy as np
from numpy import ravel_multi_index as ep
import heapq

def parse_input(fname : str) -> np.ndarray:
    with open(fname, 'r') as f:
        lines = f.readlines()

    ret_mat = np.zeros((len(lines), len(lines[0]) - 1))
    for row, l in enumerate(lines):
        l = l.strip()
        for col, k in enumerate(l):
            ret_mat[row, col] = int(k)

    return ret_mat

def default_val():
    return np.Infinity

def manhattan_dist(x : int, map_2d : np.ndarray) -> np.ndarray:
    mp_s = map_2d.shape
    end = np.array(mp_s) - 1

    x = np.unravel_index(x, mp_s)
    return np.sum(np.abs(x - end))

def reconstruct_path(came_from : dict, current_node : int, map_2d : np.ndarray) -> float:
    start = np.zeros(2, dtype=int)
    start_ravel_idx = ep(start, map_2d.shape)

    path = [current_node]
    while path[-1] != start_ravel_idx:
        path.append(came_from[path[-1]])
    return np.array(path)

def get_neighbor(ravel_idx : int, map_2d : np.ndarray) -> np.ndarray:
    '''
    Returns (4,)
    '''
    map_coords = np.array(np.unravel_index(ravel_idx, map_2d.shape))

    neighbors = np.zeros(0)
    if map_coords[0] > 0:
        up_neighbor = map_coords.copy()
        up_neighbor[0] -= 1

        neighbors = np.concatenate([neighbors, [ep(up_neighbor, map_2d.shape)]])

    if map_coords[0] < map_2d.shape[0] - 1:
        up_neighbor = map_coords.copy()
        up_neighbor[0] += 1

        neighbors = np.concatenate([neighbors, [ep(up_neighbor, map_2d.shape)]])
        
    if map_coords[1] > 0:
        up_neighbor = map_coords.copy()
        up_neighbor[1] -= 1

        neighbors = np.concatenate([neighbors, [ep(up_neighbor, map_2d.shape)]])
        
    if map_coords[1] < map_2d.shape[1] - 1:
        up_neighbor = map_coords.copy()
        up_neighbor[1] += 1

        neighbors = np.concatenate([neighbors, [ep(up_neighbor, map_2d.shape)]])
        
    return neighbors.astype(int)

def a_star(map_2d : np.ndarray) -> np.ndarray:
    start = np.zeros(2, dtype=int)
    mp_s = map_2d.shape
    end = np.array(mp_s).astype(int) - 1
    start_ravel_idx = ep(start, mp_s)
    end_ravel_idx = ep(end, mp_s)

    open_set = [(map_2d[start], start_ravel_idx)]
    heapq.heapify(open_set)

    came_from = dict()
    gScore = defaultdict(default_val)
    gScore[start_ravel_idx] = 0

    fScore = defaultdict(default_val)
    fScore[start_ravel_idx] = manhattan_dist(start_ravel_idx, map_2d)

    while len(open_set) > 0:
        (_, node_ravel_idx) = heapq.heappop(open_set)
        if node_ravel_idx == end_ravel_idx:
            return reconstruct_path(came_from, node_ravel_idx, map_2d)

        for neighbor in get_neighbor(node_ravel_idx, map_2d):
            tentative_gScore = gScore[node_ravel_idx] + map_2d[np.unravel_index(neighbor, map_2d.shape)]

            if tentative_gScore < gScore[neighbor]:
                came_from[neighbor] = node_ravel_idx
                gScore[neighbor] = tentative_gScore
                fScore[neighbor] = tentative_gScore + manhattan_dist(neighbor, map_2d)

                # TODO: O(n) look up, should fix with a parallel map look up
                # print(open_set)
                # if neighbor not in open_set:
                if not any(neighbor == i[1] for i in open_set):
                    heapq.heappush(open_set, (fScore[neighbor], neighbor))

    raise Exception("No Path Found")
    
def generate_big_map(map_2d : np.ndarray) -> np.ndarray:
    (r, c) = map_2d.shape
    big_map = np.zeros((5 * r, 5 * c))

    increment_map2d = map_2d.copy()
    big_map[:r, :c] = map_2d
    for i in range(1, 5):
        increment_map2d += 1
        increment_map2d[increment_map2d == 10] = 1
        big_map[:r, i * c : (i + 1) * c] = increment_map2d

    increment_map2d = big_map[:r, :].copy()
    for i in range(1, 5):
        increment_map2d += 1
        increment_map2d[increment_map2d == 10] = 1
        big_map[i * r : (i + 1) * r, :] = increment_map2d

    return big_map

def test_map_gen(small_map : str, big_map : str) -> bool:
    small_map = parse_input(small_map)

    big_map = parse_input(big_map)

    gen_map = generate_big_map(small_map)

    return np.allclose(big_map, gen_map)

def main():
    fname = "../myinput15"
    map2d = parse_input(fname)

    # Part 1
    result = a_star(map2d)
    path = np.unravel_index(result, map2d.shape)
    # It seems it doesn't want the starting point's cost
    print(f"P1: {np.sum(map2d[path]) - map2d[0, 0]}")

    # Part 2:
    # Generate Map
    big_map = generate_big_map(map2d)

    result = a_star(big_map)
    path = np.unravel_index(result, big_map.shape)
    print(f"P2: {np.sum(big_map[path]) - big_map[0, 0]}")

if __name__ == "__main__":
    main()

