const CI = CartesianIndex
const neighbors(coor) = [coor + c for c in (CI(0,1), CI(0,-1), CI(1,0), CI(-1,0))]
const M = fill(9, 102, 102)
M[2:101, 2:101] = mapreduce(x->parse.(Int, collect(x))', vcat, readlines("../input9"))

function walk(M, coor)
    size = 0
    todo = Set((coor, ))
    done = Set{CI}()
    while !isempty(todo)
        size += 1
        p = pop!(todo)
        push!(done, p)
        candidates = neighbors(p)
        for s in candidates
            s∉done && M[s]<9 && (push!(todo, s))
        end
    end
    return size
end

# part 1 + 2
let p1=0; p2=Int[]
for coor in CartesianIndices((2:101, 2:101))
    ns = neighbors(coor)
    if all(>(M[coor]), M[ns]) 
        p1 += M[coor]+1
        push!(p2, walk(M, coor))
    end
end
println(p1)
println(*(sort(p2)[end-2:end]...) |> sum)
end
