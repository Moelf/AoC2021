const ints = parse.(Int, readlines("./input1"))

@show count(>(0), diff(ints))
