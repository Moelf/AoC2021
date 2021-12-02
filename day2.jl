ins = readlines("./input2")

function part1_2(arr)
    x = z = aim = 0
    for l in arr
        i = parse(Int, last(l))
        if startswith(l, 'f')
            x += i
            z += i*aim
        else
            aim += ifelse(startswith(l, 'd'), i, -i)
        end
    end
    x .* (aim, z)
end
println("Day1 p1, p2: $(part1_2(ins))")
