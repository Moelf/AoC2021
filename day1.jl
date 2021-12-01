const ints = parse.(Int, readlines("./input1"))

p1 = count(>(0), diff(ints))

println("Day1: $p1")

p2 = count(4:lastindex(ints)) do idx
    ints[idx-3] < ints[idx]
end

println("Day1 p2: $p2")
