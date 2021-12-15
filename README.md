## Day 1
<details>
<summary>Julia</summary>

```julia
const ints = parse.(Int, readlines("../input1"))

p1 = sum(>(0), diff(ints))

println("Day1: $p1")

p2 = sum(4:lastindex(ints)) do idx
    ints[idx-3] < ints[idx]
end

println("Day1 p2: $p2")

```
</details>

<details>
<summary>OCaml</summary>

```ocaml
(* memory-efficient solution
 * uses continuation-passing style
 *)

(* part 1 *)
let input = open_in "../input1"
let _ = print_endline "part 1:"
(* calculates num increments as well as first depth
 * calls f num_incr first_depth
 *)
let rec k_num_incr ic f =
  match input_line ic with
        exception End_of_file -> f 0 None
      | line ->
          let this_depth = int_of_string line in
          k_num_incr ic (fun num_incr next_depth_opt ->
            match next_depth_opt with
                  None -> f 0 (Some this_depth)
                | Some next_depth ->
                    f (num_incr + if this_depth < next_depth then 1 else 0) (Some this_depth)
          )

let _ = k_num_incr input (fun num_incr _ -> (print_int num_incr; print_endline ""))

(* part 2 *)
let _ = print_endline "part 2:"
let input = open_in "../input1"
(* same idea, now storing 3 future depths instead of just 1 *)
let rec k_num_incr3 ic f =
  match input_line ic with
        exception End_of_file -> f 0 None None None
      | line ->
          let d0 = int_of_string line in
          k_num_incr3 ic
          (fun num_incr d1opt d2opt d3opt ->
            match d3opt with
                  None -> f num_incr (Some d0) d1opt d2opt
                | Some d3 -> f (num_incr + if d0 < d3 then 1 else 0) (Some d0) d1opt d2opt
          )
let _ = k_num_incr3 input (fun num_incr _ _ _ -> (print_int num_incr; print_endline ""))
                  




(* golfy solution *)
(* part 1 only *)
let input = open_in "../input1"
let _ = print_endline "part 1:"
(* reads entire file into list *)
let rec get_depths ic =
  match input_line ic with
        exception End_of_file -> []
      | line -> (int_of_string line)::(get_depths ic)

let depths = get_depths input
(* fold by keeping & updating a tuple:
 * current num increments, previous depth
 *)
let (result, _) = List.fold_left (fun (num_incr, prev_depth) this_depth -> ((num_incr + if this_depth > prev_depth then 1 else 0), this_depth)) (0, List.hd depths) (List.tl depths)

let _ = print_int result; print_endline ""


```
</details>

<details>
<summary>Python</summary>

```python
import numpy as np

def count_increment(sonar_meas : np.ndarray) -> float:
    depth_diff = np.diff(sonar_meas)
    return np.sum(depth_diff > 0)

input_depth = np.loadtxt("../input1")
print(f"Ans: {count_increment(input_depth)}")

filter_size = 3
filter = np.ones(filter_size)
rolling = np.convolve(input_depth, filter)[filter_size-1:-2]
print(f"Part2 ans: {count_increment(rolling)}")
```
</details>

<details>
<summary>Cpp</summary>

```cpp
#include <iostream>
#include <fstream>
#include <xtensor/xio.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xview.hpp>
#include <string>

int main(int argc, char** argv) {
    std::string fname = "../../input1";
    std::ifstream in_file;
    in_file.open(fname);

    xt::xarray<double> data = xt::load_csv<double>(in_file);
    xt::xarray<double> depth_diff = xt::diff(data, 1, 0);

    auto val = xt::sum(depth_diff > 0);
    std::cout << "Part 1:" << val << std::endl;

    int result = 0;
    int this_val = 0;
    for (size_t i = 0; i < data.size() - 3; i++) {
        this_val = data(i+3, 0) > data(i, 0) ? 1 : 0;
        result += this_val;
    }

    std::cout << "Part 2:" << result << std::endl;

    return 0;
}
```
</details>

## Day 2
<details>
<summary>Julia</summary>

```julia
ins = readlines("../input2")

function part1_2(arr)
    x = z = aim = 0
    for l in arr
        i = parse(Int, last(l))
        if l[1] == 'f'
            x += i
            z += i*aim
        else
            aim += ifelse(l[1] == 'd', i, -i)
        end
    end
    x .* (aim, z)
end
println("Day1 p1, p2: $(part1_2(ins))")

```
</details>

<details>
<summary>OCaml</summary>

```ocaml
let input = open_in "../input2"
let rec k_get_lines ic f =
  match input_line ic with
        exception End_of_file -> f []
      | line -> k_get_lines ic (fun ls ->
          let split = String.split_on_char ' ' line in
          f ((List.hd split, int_of_string (List.nth split 1))::ls)
        )
let lines = k_get_lines input (fun x -> x)


let forwards = List.filter (function ("forward", _) -> true | _ -> false) lines
let ups = List.filter (function ("up", _) -> true | _ -> false) lines
let downs = List.filter (function ("down", _) -> true | _ -> false) lines
let x = List.fold_left (fun total (_, dx) -> total + dx) 0 forwards
let y0 = List.fold_left (fun total (_, dy) -> total + dy) 0 ups
let y1 = List.fold_left (fun total (_, dy) -> total + dy) 0 downs
let _ = print_string "part 1: "; print_int (x * (y1 - y0)); print_endline ""




let rec calc cmds x y a =
  match cmds with
        [] -> x * y
      | ("forward", amount)::cs -> calc cs (x + amount) (y + amount * a) a
      | ("up", amount)::cs -> calc cs x y (a - amount)
      | ("down", amount)::cs -> calc cs x y (a + amount)
      | _ -> raise (Failure "bad")
let _ = print_string "part 2: "; print_int (calc lines 0 0 0); print_endline ""

```
</details>

<details>
<summary>Python</summary>

```python
import numpy as np

f_name = "../input2"
lines = None
with open(f_name, 'r') as f:
    lines = f.readlines()

# Action Mapping
action = {
    'forward' : [0, 1],
    'down' : [1, 1],
    'up' : [1, -1]
}

# Dead Reckoning
position = np.array([0, 0])
for l in lines:
    p = l.split(" ")
    action_key = p[0]
    movement = float(p[1])
    position[action[action_key][0]] += movement * action[action_key][1]

print(f"prod: {np.prod(position)}")

print("Part 2:")

# Change Action mapping since down and up affect aims now
action = {
    'forward' : [0, 1],
    'down' : [2, 1],
    'up' : [2, -1]
}

# Accordingly position aim
position_with_aim = np.array([0, 0, 0])
for l in lines:
    p = l.split(" ")
    action_key = p[0]
    movement = float(p[1])

    # Add to aim and position
    position_with_aim[action[action_key][0]] += movement * action[action_key][1]
    if action_key == "forward":
        position_with_aim[1] += movement * position_with_aim[2]

print(f"prod: {np.prod(position_with_aim[:-1])}")



```
</details>

## Day 3
<details>
<summary>Julia</summary>

```julia
rows = map(eachline("../input3")) do line
    parse.(Bool, collect(line))
end
rowidxs = eachindex(rows[1])
mode(rows, j, op = >=) = op(sum(r->r[j], rows), length(rows)÷2)
f(x) = evalpoly(2, reverse(x))

### part 1
γ = [mode(rows, j) for j in rowidxs]
ϵ = .~(γ) # bit flips
println("p1: ",  prod(f, (γ, ϵ)))

### part 2
part2(rows, j, op) = filter(r -> r[j]==mode(rows, j, op), rows)
let oxygen = rows, CO2 = rows
    for j in rowidxs
        length(oxygen)==1 && length(CO2)==1 && break
        oxygen = part2(oxygen, j, >=)
        CO2 = part2(CO2, j, <)
    end
    println("p2: ",  prod(f∘only, (oxygen, CO2)))
end

```
</details>

<details>
<summary>OCaml</summary>

```ocaml
let input = open_in "../input3"
let rec k_get_lines ic f =
  match input_line ic with
        exception End_of_file -> f []
      | line -> k_get_lines ic (fun ls -> f (line::ls))
let lines = k_get_lines input (fun x -> x)
let nbits = String.length (List.hd lines)
let bgamma = List.init nbits (fun i ->
  let zeros = List.filter (fun ln -> ln.[i] = '0') lines in
  if (2 * List.length zeros > List.length lines) then 0 else 1
)
let bepsilon = List.map (function 0 -> 1 | _ -> 0) bgamma

let rec k_rblist2dec rbls f =
  match rbls with
        [] -> f 0
      | b::rbs -> k_rblist2dec rbs (fun x -> f (2 * x + b))


let gamma = k_rblist2dec (List.rev bgamma) (fun x -> x)
let epsilon = k_rblist2dec (List.rev bepsilon) (fun x -> x)

let _ = print_string "part 1: "; print_int (gamma * epsilon); print_endline ""






let rec find_gas lines i oxygen =
  match lines with
        [] -> raise (Failure "bad")
      | [x] -> x
      | _ ->
          let zeros = List.filter (fun ln -> ln.[i] = '0') lines in
          let x =
            let more_zero = compare (2 * List.length zeros) (List.length lines) in
            if more_zero <= 0
              then (if oxygen then 1 else 0)
              else (if oxygen then 0 else 1) in
          let newlines = List.filter (fun ln -> String.sub ln i 1 = string_of_int x ) lines in
          find_gas newlines (i + 1) oxygen
let o2s = find_gas lines 0 true
let co2s = find_gas lines 0 false
let o2b = List.init (String.length o2s) (fun i -> int_of_string (String.sub o2s i 1)) 
let co2b = List.init (String.length co2s) (fun i -> int_of_string (String.sub co2s i 1)) 
let o2 = k_rblist2dec (List.rev o2b) (fun x -> x)
let co2 = k_rblist2dec (List.rev co2b) (fun x -> x)
let _ = print_string "part 2: "; print_int (o2 * co2); print_endline ""

```
</details>

<details>
<summary>Python</summary>

```python
# Too much of a havoc on the prompt side, not going to document

import numpy as np

fname = "../input3"
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

```
</details>

## Day 4
<details>
<summary>Julia</summary>

```julia
inputs = split(strip(read("../input4", String), '\n'), "\n\n")
draws = parse.(Int, split(inputs[1], ","))
boards = map(inputs[2:end]) do board
    parse.(Int, mapreduce(split, hcat, split(board, "\n")))
end
p = fill(-1,5)
wincon(m) = any(==(p), eachrow(m)) || any(==(p), eachcol(m))

# part 1 & 2
done = Set{Int}()
res = Int[]
for num in draws, (i, b) in enumerate(boards)
    replace!(b, num => -1)
    if i∉done && wincon(b)
        push!(done, i)
        push!(res, num * sum(filter(>(0), b)))
    end
end
println(res[[begin, end]])

```
</details>

<details>
<summary>OCaml</summary>

```ocaml
let input = open_in "../input4"
let rec k_get_lines ic f =
  match input_line ic with
        exception End_of_file -> f []
      | line -> k_get_lines ic (fun ls -> f (line::ls))
let lines = k_get_lines input (fun x -> x)


let nums = List.map int_of_string (String.split_on_char ',' (List.hd lines))

let line_to_row line =
  let split = String.split_on_char ' ' line in
  List.map int_of_string (List.filter (fun x -> x <> "") split)

let rec k_get_boards lines f =
  match lines with
        [] -> f []
      | line::ls -> k_get_boards ls (fun boards ->
          if line = ""
            then f ([]::boards)
            else (if boards = []
              then f [[line_to_row line]]
              else (
                let top_board = List.hd boards in
                f (((line_to_row line)::top_board)::(List.tl boards))
              )
            )
        )

let boards = k_get_boards (List.tl (List.tl lines)) (fun x -> x)
let checks = List.map (List.map (List.map (function _ -> false))) boards

let rec update_row_status row checks num =
  match row with
        [] -> []
      | x::xs -> (if x = num then true else (List.hd checks))::(update_row_status xs (List.tl checks) num)

let rec update_status board checks num =
  match board with
        [] -> checks
      | row::rows -> (if List.mem num row
          then (update_row_status row (List.hd checks) num)
          else (List.hd checks))::(update_status rows (List.tl checks) num)

let rec one_row_done one_board_checks =
  match one_board_checks with
        [] -> false
      | row::rows -> one_row_done rows || row = [true; true; true; true; true]

let one_col_done bchecks =
  let transposed = List.init 5 (fun i -> List.init 5 (fun j -> List.nth (List.nth bchecks j) i)) in
  one_row_done transposed

let board_is_done one_board_checks =
  one_row_done one_board_checks || one_col_done one_board_checks


let rec find_done_board all_checks =
  match all_checks with
        [] -> None
      | bcheck::bchecks ->
          (match find_done_board bchecks with
                 Some i -> Some (i + 1)
               | None -> if board_is_done bcheck then Some 0 else None)

let (win_checks, done_board, win_num) =
List.fold_left (fun ((checks, done_board, win_num) as state) num ->
  match done_board with
        Some _ -> state
      | None ->
          let new_checks = List.mapi (fun i one_board_check -> update_status (List.nth boards i) one_board_check num) checks in
          match find_done_board new_checks with
                None -> (new_checks, None, -1)
              | Some i -> (new_checks, Some i, num)
) (checks, None, -1) nums


let win_board = match done_board with None -> raise (Failure "bad") | Some i -> i
let win_board_checks = List.nth win_checks win_board

let rec sum_win_board board checks =
  match board with
        [] -> 0
      | row::rows -> List.fold_left (+) 0 (List.mapi (fun i x -> if List.nth (List.hd checks) i then 0 else x) row) + sum_win_board rows (List.tl checks)

let calc_win_score board checks num =
  let sum_num = sum_win_board board checks in
  sum_num * num

let _ = print_string "part 1: "; print_int (calc_win_score (List.nth boards win_board) win_board_checks win_num); print_endline ""


let (_, last_score) =
List.fold_left (fun (boards_and_checks, last_score) num ->
  let new_bc = List.map (fun (board, check) ->
    (board, update_status board check num)
  ) boards_and_checks in
  let not_done = List.filter (fun (b, c) -> not (board_is_done c)) new_bc in
  let done_stuff = List.filter (fun (b, c) -> (board_is_done c)) new_bc in
  let (done_boards, done_boards_checks) = List.split done_stuff in
  if done_stuff <> []
    then (not_done, calc_win_score (List.nth done_boards (List.length done_boards - 1)) (List.nth done_boards_checks (List.length done_boards - 1)) num)
    else (not_done, last_score)
) (List.combine boards checks, -1) nums


let _ = print_string "part 2: "; print_int last_score; print_endline ""

```
</details>

<details>
<summary>Python</summary>

```python
import numpy as np
import tempfile

# Read File input
file_name = "../input4"
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

```
</details>

## Day 5
<details>
<summary>Julia</summary>

```julia
using LinearAlgebra
CI = CartesianIndex
_sign(x1,x2) = x1>=x2 ? -1 : 1

lines = split.(readlines("../input5"), r",| -> ")
coords = map(lines) do line
    c = parse.(Int, line)
    step = CI(_sign(c[1], c[3]), _sign(c[2], c[4]))
    CI(c[1], c[2]):step:CI(c[3], c[4])
end
# part 1
M = zeros(Int, 1000, 1000)
for c in coords
    any(==(1), size(c)) && (M[c] .+= 1)
end
println("P1: ", sum(>=(2), M))
# part 2
for c in coords
    any(==(1), size(c)) || (M[diag(c)] .+= 1)
end
println("P2: ", sum(>=(2), M))

```
</details>

<details>
<summary>OCaml</summary>

```ocaml
let input = open_in "../input5"
let rec k_get_lines ic f =
  match input_line ic with
        exception End_of_file -> f []
      | line -> k_get_lines ic (fun ls ->
          let intlist = (List.map int_of_string (Str.split (Str.regexp ",\\| -> ") line)) in
          match intlist with
                [a; b; c; d] -> f ((a, b, c, d)::ls)
              | _ -> raise (Failure "bad")
        )
let lines = k_get_lines input (fun x -> x)

let grid = Array.make_matrix 1000 1000 0

let rec update_col grid x0 x1 y =
  if x0 > x1 then () else
  let row = Array.get grid x0 in
  let _ = Array.set row y (1 + Array.get row y) in
  update_col grid (x0 + 1) x1 y

let rec update_row row y0 y1 =
  if y0 > y1 then () else
  let _ = Array.set row y0 (1 + Array.get row y0) in
  update_row row (y0 + 1) y1

let rec update_diag_up grid (x0, y0, x1, y1) =
  if x0 > x1 then () else
  let row = Array.get grid x0 in
  let _ = Array.set row y0 (1 + Array.get row y0) in
  update_diag_up grid (x0 + 1, y0 - 1, x1, y1)

let rec update_diag_down grid (x0, y0, x1, y1) =
  if x0 > x1 then () else
  let row = Array.get grid x0 in
  let _ = Array.set row y0 (1 + Array.get row y0) in
  update_diag_down grid (x0 + 1, y0 + 1, x1, y1)

let rec update_diag grid (x0, y0, x1, y1) =
  if x0 > x1 then update_diag grid (x1, y1, x0, y0) else
  if y0 > y1
    then update_diag_up grid (x0, y0, x1, y1)
    else update_diag_down grid (x0, y0, x1, y1)

let update_grid grid (x0, y0, x1, y1) =
  if x0 = x1
    then if y0 > y1 then update_row (Array.get grid x0) y1 y0 else update_row (Array.get grid x0) y0 y1
    else if x0 < x1 then update_col grid x0 x1 y0 else update_col grid x1 x0 y0

let _ = List.iter (fun ((x0, y0, x1, y1) as line) ->
  if x0 <> x1 && y0 <> y1
    then update_diag grid line
    else update_grid grid line
) lines

let count = Array.fold_left (fun count row -> count + Array.fold_left (fun c x -> c + if x > 1 then 1 else 0) 0 row) 0 grid

```
</details>

<details>
<summary>Python</summary>

```python
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

fname = "../input5"
observation = parse_input(fname)
empty_map = create_map_from_input(observation)
filled_map = map_world(empty_map.copy(), observation)
print(f"Part1: {count_overlap(filled_map, 2)}")

part2_filled_map = map_world(empty_map.copy(), observation, part1=False)
print(f"Part2: {count_overlap(part2_filled_map, 2)}")


```
</details>

## Day 6
<details>
<summary>Julia</summary>

```julia
using LinearAlgebra
const ary = parse.(Int, split(readline("../input6"), ","))

function f(ary, days)
    counts = zeros(Int, 9)
    for a in ary
        counts[a+1] += 1
    end
    ker = Int[0 0 0 0 0 0 1 0 1
        I(8) zeros(8)]
    sum(counts' * ker^days)
end

println("P1: $(f(ary, 80)), P2: $(f(ary, 256))")

```
</details>

<details>
<summary>OCaml</summary>

```ocaml
let list_set l n x =
  List.init (List.length l) (fun i -> if i = n then x else List.nth l i)

let grow count =
  match count with [] -> raise (Failure "bad") |
        zero::rest -> (list_set rest 6 (List.nth rest 6 + zero)) @ [zero]

let rec grow_days count days =
  match days with
        0 -> count
      | _ -> grow_days (grow count) (days - 1)






let input_path = "../input6"
let days = 256
let fish = List.map int_of_string (String.split_on_char ',' (input_line (open_in input_path)))
let count = List.fold_left (fun c f -> list_set c f (1 + List.nth c f)) (List.init 9 (fun _ -> 0)) fish
let end_count = grow_days count days
let total = List.fold_left (+) 0 end_count

```
</details>

<details>
<summary>Python</summary>

```python
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

fname = "../input6"
state = parse_input(fname)

part1_days, part2_days = 80, 256
part1_ans = particle_methods(state.copy(), part1_days)
print(f"{part1_days} days of fish: {part1_ans}")

part2_ans = binning_method(state.copy(), part2_days)
print(f"{part2_days} days of fish: {part2_ans}")

```
</details>

## Day 7
<details>
<summary>Julia</summary>

```julia
const ints = parse.(Int, split(readline("../input7"), ',')) |> sort
println("p1: ", sum(abs, ints .- ints[end÷2]) |> Int)

cost(x) = sum(1:abs(x))
println("p2: ", minimum(sum(cost, ints .- idx) for idx in 0:maximum(ints)))

```
</details>

<details>
<summary>OCaml</summary>

```ocaml
let get2fuel x y =
  let d = abs (x - y) in
  (1 + d) * d / 2



(* part 1 is getting median *)
let input_path = "./input7"
let crabs = List.sort compare (List.map int_of_string (String.split_on_char ',' (input_line (open_in input_path))))
let middle = (List.length crabs) / 2
let middle_fuel = List.fold_left (fun total crab -> total + abs (crab - List.nth crabs middle)) 0 crabs






(* part 2 is brute force search *)
let fuels2 = List.hd (List.sort compare (List.map (fun pos -> List.fold_left (fun fuel crab -> fuel + get2fuel crab pos) 0 crabs) (List.init (List.nth crabs (List.length crabs - 1)) (fun i -> i + List.hd crabs))))

```
</details>

<details>
<summary>Python</summary>

```python
import numpy as np
from typing import Any, Callable
from numbers import Number

def cost_p1(crab_loc : np.ndarray, loc : float) -> float:
    return np.sum(np.abs(crab_loc - loc))

def cost_p2(crab_loc : np.ndarray, loc : Any) -> float:
    if isinstance(loc, Number):
        loc = np.array([loc])
    diff = np.abs(crab_loc[:, np.newaxis] - loc[np.newaxis, :])
    return np.sum((1 + diff) * diff / 2, axis=0)

def grad_p1(crab_loc : np.ndarray, soln : float) -> float:
    return np.mean(np.sign(crab_loc - soln))

def min_hori_dist(crab_loc : np.ndarray, grad_fn : Callable) -> float:
    '''
    Momentum to account for L1 non-differentiability :)
    '''

    cont_soln = np.mean(crab_loc) # L2 solution 
    grad = 0
    last_grad = 0
    gamma = 0.9

    max_iter = 1000
    for i in range(max_iter):
        curr_grad = grad_fn(crab_loc, cont_soln)
        grad = grad * gamma + curr_grad
        cont_soln = cont_soln + grad

        if i > 50:
            # Early Termination
            if np.sign(last_grad) != np.sign(grad):
                break
        last_grad = grad

    return np.round(cont_soln)

def part2(crab_loc : np.ndarray) -> float:
    min_d = np.mean(crab_loc) # Exact solution, since it's a L2 cost now
    d = np.array([np.ceil(min_d), np.floor(min_d)])
    return np.min(cost_p2(crab_loc, d))

fname = "../input7"
crab_loc = np.loadtxt(fname, delimiter=",")
min_loc = min_hori_dist(crab_loc, grad_p1)
print(f"Part1 index: {min_loc}, cost: {cost_p1(crab_loc, 362)}")
print(f"Part2 cost: {part2(crab_loc)}")

```
</details>

## Day 8
<details>
<summary>Julia</summary>

```julia
const lines = split.(readlines("../input8"), r" \| | ")

# part 1
p1 = sum(lines) do line
    sum(x -> length(x)∈(2,4,3,7), line[end-3:end])
end
println("P1: ", p1)

# part 2
# one-liner
println(sum(parse.(Int,["4725360918"[[sum([L...].∈r)÷2%15%11+1 for r in split(R)]] 
for (L,R) in split.(readlines("../input8"),'|')])))

# you can read this off from the standard segments pattern
standard_patterns = ["abcefg", "cf", "acdeg", "acdfg", "bcdf", 
    "abdfg", "abdefg", "acf", "abcdefg", "abcdfg"]

# make a scoremap for how many times each segments show up in 0-9 pattern
scoremap = Dict(x => sum(==(x), standard_patterns |> join) for x in 'a':'g')

# each pattern can be uniquely determined by summing the lit segments' score
# here we make a standard score look up table with known segments mapping
const standards = map(standard_patterns) do s
    sum(scoremap[c] for c in s)
end

function disam(patterns, output)
    # make score for each new set of patterns
    _scores = Dict(x => sum(==(x), patterns |> join) for x in 'a':'g')
    
    # get the sum of segments for the four output digits
    _digits = map(output) do s
        sum(_scores[c] for c in s)
    end
    # look up corresponding digits in the standard pattern (by index)
    # offset by one because 0/1 indexing...
    [findfirst(==(x), standards) - 1 for x in _digits]
end

p2 = sum(lines) do line
    patterns = line[1:10]
    output = line[11:end]
    res = disam(patterns, output)
    evalpoly(10, reverse(res))
end

```
</details>

<details>
<summary>Python</summary>

```python
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

```
</details>

## Day 9
<details>
<summary>Julia</summary>

```julia
const CI = CartesianIndex
const neighbors(coor) = [coor + c for c in (CI(0,1), CI(0,-1), CI(1,0), CI(-1,0))]
const M = fill(9, 102, 102)
M[2:101, 2:101] = mapreduce(x->parse.(Int, collect(x))', vcat, readlines("../input9"))

function walk(M, coor)
    size = 0
    todo = Set((coor, ))
    done = Set{CI}()
    while !isempty(todo)
        size += 1
        p = pop!(todo)
        push!(done, p)
        candidates = neighbors(p)
        for s in candidates
            s∉done && M[s]<9 && (push!(todo, s))
        end
    end
    return size
end

# part 1 + 2
let p1=0; p2=Int[]
    for coor in CartesianIndices((2:101, 2:101))
        ns = neighbors(coor)
        if all(>(M[coor]), M[ns]) 
            p1 += M[coor]+1
            push!(p2, walk(M, coor))
        end
    end
    println(p1)
    println(*(sort(p2)[end-2:end]...) |> sum)
end

```
</details>

<details>
<summary>Python</summary>

```python
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

```
</details>

## Day 10
<details>
<summary>Julia</summary>

```julia
function checker(line, part2)
    res1 = res2 = 0
    match = Dict('(' => ')', '[' => ']', '{' => '}', '<' => '>')
    smap =  Dict('(' => 1,   '[' => 2,   '{' => 3,   '<' => 4,
                 ')' => 3,   ']' => 57,  '}' => 1197,'>' => 25137)
    stack = Char[]
    for t in line
        if t ∈ "([{<"
            push!(stack, t)
        elseif t == match[last(stack)]
            pop!(stack)
        else
            res1 = smap[t]
            @goto corrupted
        end
    end
    foreach(reverse(stack)) do r
        res2 *= 5
        res2 += smap[r]
    end
    push!(part2, res2)
    @label corrupted
    res1
end

const p2 = Int[]
const ls = readlines("../input10")
L(l) = checker(l, p2)
empty!(p2)
println("P1: ", sum(L, ls))
println("P2: ", sort(p2)[end ÷ 2 + 1])
@time sum(L, ls)
@time sort(p2)[end ÷ 2 + 1]

```
</details>

<details>
<summary>Python</summary>

```python
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

```
</details>

## Day 11
<details>
<summary>Julia</summary>

```julia
CI = CartesianIndex
const M = mapreduce(l->parse.(Int, collect(l))', vcat, eachline("../input11"))
const M2 = copy(M)

const adj = setdiff(CI(-1,-1):CI(1,1), (CI(0,0), ))
flash!(M, CM, c) = [M[c+a]+=1 for a in adj if c+a ∈ CM]

function step!(M)
    CM = CartesianIndices(M)
    M .+= 1
    @label again
    for c in CM
        (M[c] > 20 || M[c] < 10) && continue
        M[c] = 30
        flash!(M, CM, c)
        @goto again
    end
    M[M .> 9] .= 0
    sum(iszero, M)
end

println("p1: ", sum(x->step!(M), 1:100))
println("p2: ", findfirst(x->step!(M2)==100, 1:1000))

```
</details>

## Day 12
<details>
<summary>Julia</summary>

```julia
const pairs = [=>(split(x, '-')...) for x in eachline("../input12")]

function islegal(name, path, twice)
    name == "start" && return false
    if name ∉ path || all(isuppercase, name)
        return true
    elseif twice[]
        twice[] = false
        return true
    end
    return false
end

function explore(path=["start"]; res=[0], twice=[false])
    here = last(path)
    if here == "end" 
        res[] += 1
        return res[]
    end
    for p in pairs
        t1 = copy(twice); t2 = copy(twice)
        p[1] == here && islegal(p[2], path, t1) && explore([path; p[2]]; res, twice=t1)
        p[2] == here && islegal(p[1], path, t2) && explore([path; p[1]]; res, twice=t2)
    end
    res[]
end

println("P1 :", explore())
println("P2 :", explore(twice=[true]))

```
</details>

## Day 13
<details>
<summary>Julia</summary>

```julia
using SparseArrays

raw_p, raw_i = strip.(split(read("../input13", String), "\n\n"))
points = [parse.(Int, (y,x)) .+ 1 for (x,y) in split.(split(raw_p, "\n"), ",")]
instructions = split(raw_i, "\n")

let M = sparse(first.(points), last.(points), true)
    for (i, line) in enumerate(instructions)
        @show size(M)
        M = if contains(line, "y=")
            M[1:end÷2, :] .|| M[end:-1:end÷2+2, :]
        else
            M[:, 1:end÷2] .|| M[:, end:-1:end÷2+2]
        end
        i == 1 && println("P1: ", sum(!iszero, M))
    end
    display(M)
end

```
</details>

## Day 14
<details>
<summary>Julia</summary>

```julia
raw_t, _, raw_rules... = readlines("../input14")
const rules = Dict(=>(x...) for x in split.(raw_rules, " -> "))
add!(dict, key, val) = dict[key] = val + get(dict, key, 0)

let polymer = Dict("$x"=>count(==(x), raw_t) for x in raw_t)
    local atoms
    temp = Dict(k=>0 for k in keys(polymer))
    for i in 1:lastindex(raw_t)-1 # initial pairs
        polymer[raw_t[i:i+1]] = 1
    end
    # Part 1 + 2
    for i = 1:40
        map!(zero, values(temp)) # clear buffer
        for (pair, insertion) in rules
            left, right = pair[1]*insertion, insertion*pair[2]
            Npair = get(polymer, pair, 0)
            add!.(Ref(temp), (left, insertion, right), Npair)
            add!(temp, pair, -Npair) # breaking pairs
        end
        add!.(Ref(polymer), keys(temp), values(temp)) # use buffer
        atoms = sort([v for (k,v) in polymer if length(k) == 1])
        i == 10 && println("P1: ", last(atoms) - first(atoms))
    end
    println("P2: ", last(atoms) - first(atoms))
end

```
</details>

## Day 15
<details>
<summary>Julia</summary>

```julia
const M = parse.(Int, mapreduce(collect, hcat, eachline("../input15")))
const M2 = hvncat((5,5), false, [@. mod1(M+i+j, 9) for i=0:4, j=0:4]...)
const CI = CartesianIndex
const adjs = Tuple(CI(x) for x in ((0,1), (0,-1), (1,0), (-1,0)))

function main(M)
    CM = CartesianIndices(M)
    Q = [0=>CI(1,1)]
    tot_risks = fill(typemax(Int), size(M))
    while true
        risk, here = pop!(Q)
        tot_risks[here] <= risk && continue # not a better route
        tot_risks[here] = risk
        here == last(CM) && return risk # reached goal

        for a in adjs
            (ne = here+a) ∉ CM && continue
            s = risk + M[ne] => ne
            i = searchsortedfirst(Q, s; rev=true)
            insert!(Q, i, s)
        end
    end
end
println("P1: ", main(M))
println("P2: ", main(M2))

```
</details>

<details>
<summary>Python</summary>

```python
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


```
</details>

## Day 16
## Day 17
## Day 18
## Day 19
## Day 20
## Day 21
## Day 22
## Day 23
## Day 24
## Day 25
## Day 26
## Day 27
## Day 28
## Day 29
## Day 30
