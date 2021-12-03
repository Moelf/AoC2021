(* calls f num_incr ic_head *)
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

let input = open_in "./input1"
let _ = k_num_incr input (fun num_incr _ -> (print_int num_incr; print_endline ""))

