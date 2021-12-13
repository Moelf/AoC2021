from typing import Tuple
import numpy as np
from sklearn.cluster import DBSCAN # :)

def parse_input(fname : str) -> np.ndarray:
    with open(fname, 'r') as f:
        lines = f.readlines()

    ret_mat = np.zeros((len(lines), len(lines[0]) - 1))
    for row, l in enumerate(lines):
        l = l.strip()
        for col, k in enumerate(l):
            ret_mat[row, col] = int(k)

    return ret_mat

def find_low_point(map_2d : np.ndarray) -> np.ndarray:
    hori_diff = np.diff(map_2d, axis=1)
    vert_diff = np.diff(map_2d, axis=0)

    saddle_map = np.ones_like(map_2d, dtype=bool)
    saddle_map[:, :-1] = hori_diff > 0
    saddle_map[:, 1:] = np.logical_and(saddle_map[:, 1:], hori_diff < 0)

    saddle_map[:-1, :] = np.logical_and(saddle_map[:-1, :], vert_diff > 0)
    saddle_map[1:, :] = np.logical_and(saddle_map[1:, :], vert_diff < 0)

    return saddle_map

def part1_sum(locations : Tuple[np.ndarray], map_2d : np.ndarray) -> float:
    return np.sum(map_2d[locations]) + len(locations[0])

def part2_cluster(map_2d : np.ndarray, largest_n : int = 3) -> float:
    map_2d[map_2d == 0] = 1
    map_2d[map_2d == 9] = 0
    coords = np.vstack(np.nonzero(map_2d)).T

    db = DBSCAN(eps=1.1, min_samples=3).fit(coords)
    alter = np.histogram(db.labels_, np.unique(db.labels_))[0]
    return np.sort(alter)[-largest_n:]

def main():
    fname = "../input9"
    map_2d = parse_input(fname)
    saddle_map = find_low_point(map_2d)
    print(f"part1: {part1_sum(np.nonzero(saddle_map),map_2d)}")
    val = part2_cluster(map_2d, 3)
    print(f"Part2: {np.prod(val)}")
    
if __name__ == "__main__":
    main()
