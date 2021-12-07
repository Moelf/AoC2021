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
