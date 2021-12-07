rows = map(eachline("../input3")) do line
    parse.(Bool, collect(line))
end
rowidxs = eachindex(rows[1])
mode(rows, j, op = >=) = op(sum(r->r[j], rows), length(rows)÷2)
f(x) = evalpoly(2, reverse(x))

### part 1
γ = [mode(rows, j) for j in rowidxs]
ϵ = .~(γ) # bit flips
println("p1: ",  prod(f, (γ, ϵ)))

### part 2
part2(rows, j, op) = filter(r -> r[j]==mode(rows, j, op), rows)
let oxygen = rows, CO2 = rows
    for j in rowidxs
        length(oxygen)==1 && length(CO2)==1 && break
        oxygen = part2(oxygen, j, >=)
        CO2 = part2(CO2, j, <)
    end
    println("p2: ",  prod(f∘only, (oxygen, CO2)))
end
