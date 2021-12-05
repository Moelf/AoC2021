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
