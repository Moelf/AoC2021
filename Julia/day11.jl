CI = CartesianIndex
const M = mapreduce(l->parse.(Int, collect(l))', vcat, eachline("../input11"))

const adj = setdiff(CI(-1,-1):CI(1,1), (CI(0,0), ))
flash!(M, CM, c) = [M[c+a]+=1 for a in adj if c+a ∈ CM]

function step!(M)
    CM = CartesianIndices(M)
    M .+= 1
    flashed = 0
    while true
        for c in CM
            if 20 > M[c] > 9
                M[c] = 30
                flashed += 1
                flash!(M, CM, c)
            end
        end
        any(x-> 20>x>9, M) || break
    end
    M[M .> 9] .= 0
    sum(iszero, M)
end

M2 = copy(M)
println("p1: ", sum(x->step!(M), 1:100))
println("p2: ", findfirst(x->step!(M2)==100, 1:1000))
