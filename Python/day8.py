from typing import Dict, List
import numpy as np

num_segment = {
    0 : 6,
    1 : 2,
    2 : 5,
    3 : 5,
    4 : 4,
    5 : 5,
    6 : 6,
    7 : 3,
    8 : 7,
    9 : 6
}

def sort_str(in_str  : str) -> str:
    return "".join(sorted(in_str))

def gen_decode_map(in_str : str) -> Dict:
    '''
    '''
    
    entries = in_str.split(" ")
    sorted_entry = sorted(entries, key=len)
    
    one = sort_str(sorted_entry[0])
    four = sort_str(sorted_entry[2])
    
    ret_entry = dict()
    ret_entry[one] = 1
    ret_entry[four] = 4
    ret_entry[sort_str(sorted_entry[1])] = 7
    ret_entry[sort_str(sorted_entry[-1])] = 8
    
    # Of length 6
    six_char_cand = [sort_str(sorted_entry[-2]),
                     sort_str(sorted_entry[-3]), sort_str(sorted_entry[-4])]
   
    mask = np.array([0, 0, 0])
    nine_str = None
    for idx, i in enumerate(six_char_cand):
        if (four[0] in i) and (four[1] in i) and (four[2] in i) and (four[3] in i):
            ret_entry[i] = 9
            nine_str = i
            mask[idx] = 1
            continue
            
        if (one[0] in i) and (one[1] in i) and i not in ret_entry:
            ret_entry[i] = 0
            mask[idx] = 1
            continue
    
    ret_entry[six_char_cand[np.argmin(mask)]] = 6

    five_char_cand = [sort_str(sorted_entry[3]),
                     sort_str(sorted_entry[4]), sort_str(sorted_entry[5])]
    
    for idx, c in enumerate(five_char_cand):
        if (one[0] in c) and (one[1] in c):
            ret_entry[c] = 3
            continue
        
        five_in_nine = True
        for s in c:
            five_in_nine = five_in_nine and (s in nine_str)
        if five_in_nine:
            ret_entry[c] = 5
            continue

        ret_entry[c] = 2

    return ret_entry

def decode_str(in_str : str, decode_map : Dict[str, int]) -> int:
    '''
        are you happy with one liners :)
    '''
    nums = [int(decode_map[sort_str(s)]) * (10 ** (3 - i)) for i, s in enumerate(in_str.split(" "))]
    return np.sum(nums)

def count_unique_nums(input: List[str]) -> int:
    '''
    '''
    unique_set = {2, 4, 3, 7}
    ret_num = 0
    for line in input:
        out_str = line.split(" ")
        k = [len(digit) in unique_set for digit in out_str]
        ret_num += np.sum(np.array(k))
    return ret_num
    
def main(fname : str = "../input8"):

    with open(fname, 'r') as f:
        line = f.readlines()

    part1_lines = [l.strip().split(" | ")[1] for l in line]

    print(f"P1: {count_unique_nums(part1_lines)}")

    part2_lines = [l.strip().split(" | ")[0] for l in line]
    ret_val = 0
    for idx, p2 in enumerate(part2_lines):
        decode_map = gen_decode_map(p2)
        num = decode_str(part1_lines[idx], decode_map)
        ret_val += num
    print(f"P2: {ret_val}")
        
if __name__ == "__main__":
    main()
