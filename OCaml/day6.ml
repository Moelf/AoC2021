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
