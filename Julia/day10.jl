function checker(line, part2)
    res1 = res2 = 0
    match = Dict('(' => ')', '[' => ']', '{' => '}', '<' => '>')
    smap =  Dict('(' => 1,   '[' => 2,   '{' => 3,   '<' => 4,
                 ')' => 3,   ']' => 57,  '}' => 1197,'>' => 25137)
    stack = Char[]
    for t in line
        if t ∈ "([{<"
            push!(stack, t)
        elseif t == match[last(stack)]
            pop!(stack)
        else
            res1 = smap[t]
            @goto corrupted
        end
    end
    foreach(reverse(stack)) do r
        res2 *= 5
        res2 += smap[r]
    end
    push!(part2, res2)
    @label corrupted
    res1
end

p2 = Int[]
println("P1: ", sum(l -> checker(l, p2), eachline("../input10")))
println("P2: ", sort(p2)[end ÷ 2 + 1])
