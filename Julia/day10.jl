function checker(line, part2)
    match = Dict('(' => ')', '[' => ']', '{' => '}', '<' => '>')
    smap = Dict('(' => 1, '[' => 2, '{' => 3, '<' => 4,
        ')' => 3, ']' => 57, '}' => 1197, '>' => 25137)
    record = Char[]
    res1 = res2 = 0
    for t in line
        if t ∈ "([{<"
            push!(record, t)
        elseif t == match[last(record)]
            pop!(record)
        else
            res1 = smap[t]
            @goto hack
        end
    end
    foreach(reverse(record)) do r
        res2 *= 5
        res2 += smap[r]
    end
    push!(part2, res2)
    @label hack
    res1
end

p2 = Int[]
t = map(l->checker(l, p2), eachline("../input10"))
println("P1: ", sum(t))
println("P2: ", sort(p2)[end÷2+1])
