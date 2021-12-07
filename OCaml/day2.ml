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
