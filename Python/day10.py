from typing import List
import numpy as np

def isValid(s : str) -> bool:
    correspondence = {
        "(" : ")",
        "[" : "]",
        "{" : "}",
        "<" : ">"
    }

    left = {
        "(",
        "[",
        "{",
        "<"
    }

    score = {
        ")" : 3,
        "]" : 57,
        "}" : 1197,
        ">" : 25137
    }

    book_keeping = list()

    for idx, char in enumerate(s):
        if char in left:
            book_keeping.append(char)
        else:
            if len(book_keeping) == 0:
                return score[char], book_keeping
            
            val = book_keeping.pop()
            if correspondence[val] != char:
                return score[char], book_keeping
    
    return 0, book_keeping

def reconstruct(l_c : List[str]) -> str:
    correspondence = {
        "(" : ")",
        "[" : "]",
        "{" : "}",
        "<" : ">"
    }

    score = {
        ")" : 1,
        "]" : 2,
        "}" : 3,
        ">" : 4
    }

    ret_str = 0
    for s in reversed(l_c):
        ret_str = ret_str * 5 + score[correspondence[s]]
    return ret_str

def part1(fname : str) -> int:
    with open(fname, 'r') as f:
        lines = f.readlines()
    syn_err = 0
    for l in lines:
        syn_err += isValid(l.strip())[0]
    return syn_err

def part2(fname : str) -> int:
    with open(fname, 'r') as f:
        lines = f.readlines()
    
    v = np.zeros(0)
    for l in lines:
        res = isValid(l.strip())
        remaining_str = res[1]
        if res[0] <= 0:
            score = reconstruct(remaining_str)
            v = np.concatenate((v, [score]))
    return v

def main():
    fname = "../myinput10"
    syn_err = part1(fname)
    print(f"Part1: {syn_err}")
    val = part2(fname)
    sorted_val = np.sort(val)
    print(np.median(sorted_val))

if __name__ == "__main__":
    main()
