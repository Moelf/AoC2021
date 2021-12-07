const ints = parse.(Int, split(readline("./input7"), ',')) |> sort
println("p1: ", sum(abs, ints .- ints[end÷2]) |> Int)

cost(x) = sum(1:abs(x))
println("p2: ", minimum(sum(cost, ints .- idx) for idx in 0:maximum(ints)))
