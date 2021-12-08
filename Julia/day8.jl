const lines = split.(readlines("../input8"), r" \| | ")

# part 1
p1 = sum(lines) do line
    sum(x -> length(x)∈(2,4,3,7), line[end-3:end])
end
println("P1: ", p1)

# part 2
# you can read this off from the standard segments pattern
standard_patterns = ["abcefg", "cf", "acdeg", "acdfg", "bcdf", 
    "abdfg", "abdefg", "acf", "abcdefg", "abcdfg"]

# make a scoremap for how many times each segments show up in 0-9 pattern
scoremap = Dict(x => sum(==(x), standard_patterns |> join) for x in 'a':'g')

# each pattern can be uniquely determined by summing the lit segments' score
# here we make a standard score look up table with known segments mapping
const standards = map(standard_patterns) do s
    sum(scoremap[c] for c in s)
end

function disam(patterns, output)
    # make score for each new set of patterns
    _scores = Dict(x => sum(==(x), patterns |> join) for x in 'a':'g')
    
    # get the sum of segments for the four output digits
    _digits = map(output) do s
        sum(_scores[c] for c in s)
    end
    # look up corresponding digits in the standard pattern (by index)
    # offset by one because 0/1 indexing...
    [findfirst(==(x), standards) - 1 for x in _digits]
end

p2 = sum(lines) do line
    patterns = line[1:10]
    output = line[11:end]
    res = disam(patterns, output)
    evalpoly(10, reverse(res))
end
println("P2: ", p2)
