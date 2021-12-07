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
