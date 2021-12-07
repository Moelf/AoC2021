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
