## Day 1
<details>
<summary>Julia</summary>

```julia
const ints = parse.(Int, readlines("./input1"))

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
let input = open_in "./input1"
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
let input = open_in "./input1"
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
let input = open_in "./input1"
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

input_depth = np.loadtxt("input1")
print(f"Ans: {count_increment(input_depth)}")

filter_size = 3
filter = np.ones(filter_size)
rolling = np.convolve(input_depth, filter)[filter_size-1:-2]
print(f"Part2 ans: {count_increment(rolling)}")
```
</details>

## Day 2
<details>
<summary>Julia</summary>

```julia
ins = readlines("./input2")

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
let input = open_in "./input2"
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

f_name = "input2"
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
rows = map(eachline("./input3")) do line
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
let input = open_in "./input3"
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

fname = "input3"
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
inputs = split(strip(read("./input4", String), '\n'), "\n\n")
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
let input = open_in "./input4"
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

```
</details>

## Day 5
<details>
<summary>Julia</summary>

```julia
using LinearAlgebra
CI = CartesianIndex
_sign(x1,x2) = x1>=x2 ? -1 : 1

lines = split.(readlines("./input5"), r",| -> ")
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
let input = open_in "./input5"
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

fname = "input5"
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
const ary = parse.(Int, split(readline("./input6"), ","))

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






let input_path = "./input6"
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

fname = "input6"
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
const ints = parse.(Int, split(readline("./input7"), ',')) |> sort
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

fname = "input7"
crab_loc = np.loadtxt(fname, delimiter=",")
min_loc = min_hori_dist(crab_loc, grad_p1)
print(f"Part1 index: {min_loc}, cost: {cost_p1(crab_loc, 362)}")
print(f"Part2 cost: {part2(crab_loc)}")

```
</details>

