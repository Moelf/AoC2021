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

const p2 = Int[]
const ls = readlines("../input10")
L(l) = checker(l, p2)
empty!(p2)
println("P1: ", sum(L, ls))
println("P2: ", sort(p2)[end ÷ 2 + 1])
@time sum(L, ls)
@time sort(p2)[end ÷ 2 + 1]
