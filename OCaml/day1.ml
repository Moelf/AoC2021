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

