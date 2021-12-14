raw_t, _, raw_rules... = readlines("../input14")
const rules = Dict(=>(x...) for x in split.(raw_rules, " -> "))
add!(dict, key, val) = dict[key] = val + get(dict, key, 0)

let polymer = Dict("$x"=>count(==(x), raw_t) for x in raw_t)
    local atoms
    for i in 1:lastindex(raw_t)-1
        polymer[raw_t[i:i+1]] = 1
    end
    # Part 1 + 2
    temp = Dict(k=>0 for k in keys(polymer))
    for i = 1:40
        map!(zero, values(temp)) # clear buffer
        for (pair, insertion) in rules
            left, right = pair[1]*insertion, insertion*pair[2]
            Npair = get(polymer, pair, 0)

            add!(temp, pair, -Npair) # breaking pairs
            for i in (insertion, left, right)
                add!(temp, i, Npair)
            end
        end

        for k in keys(temp)
            add!(polymer, k, temp[k])
        end

        atoms = sort([v for (k,v) in polymer if length(k) == 1])
        i == 10 && println("P1: ", last(atoms) - first(atoms))
    end
    println("P2: ", last(atoms) - first(atoms))
end
