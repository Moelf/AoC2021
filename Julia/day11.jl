CI = CartesianIndex
const M = mapreduce(l->parse.(Int, collect(l))', vcat, eachline("../input11"))
const M2 = copy(M)

const adj = setdiff(CI(-1,-1):CI(1,1), (CI(0,0), ))
flash!(M, CM, c) = [M[c+a]+=1 for a in adj if c+a ∈ CM]

function step!(M)
    CM = CartesianIndices(M)
    M .+= 1
    @label again
    for c in CM
        (M[c] > 20 || M[c] < 10) && continue
        M[c] = 30
        flash!(M, CM, c)
        @goto again
    end
    M[M .> 9] .= 0
    sum(iszero, M)
end

println("p1: ", sum(x->step!(M), 1:100))
println("p2: ", findfirst(x->step!(M2)==100, 1:1000))
