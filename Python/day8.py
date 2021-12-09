from typing import Dict
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

fname = "../input8"

with open(fname, 'r') as f:
    line = f.readline()

decode, output = line.split(" | ")

def sort_str(in_str  : str) -> str:
    return "".join(sorted(in_str))

def decode_str(in_str : str) -> Dict:
    '''
    '''
    
    entries = in_str.split(" ")
    sorted_entry = sorted(entries, key=len)
    
    one = sort_str(sorted_entry[0])
    four = sort_str(sorted_entry[2])
    
    ret_entry = dict()
    ret_entry[one]=1
    ret_entry[four]=4
    ret_entry[sort_str(sorted_entry[1])]=7
    ret_entry[sort_str(sorted_entry[-1])] = 8
    
    six_char_cand = [sort_str(sorted_entry[-2]),
                     sort_str(sorted_entry[-3]), sort_str(sorted_entry[-4])]
    
    mask = np.array([0, 0, 0])
    for idx, i in enumerate(six_char_cand):
        if (four[0] in i) and (four[1] in i) and (four[2] in i) and (four[3] in i):
            ret_entry[i] = 9
            mask[idx] = 1
            
        if (one[0] in i) and (one[1] in i):
            ret_entry[i] = 0
            mask[idx] = 1
    
    ret_entry[six_char_cand[np.argmin(mask)]] = 6
    

    
    
decode_str(decode)
