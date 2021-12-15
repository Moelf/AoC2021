const M = parse.(Int, mapreduce(collect, hcat, eachline("../input15")))
const M2 = hvncat((5,5), false, [@. mod1(M+i+j, 9) for i=0:4, j=0:4]...)
const CI = CartesianIndex
const adjs = [CI(x) for x in ((0,1), (0,-1), (1,0), (-1,0))]

function main(M)
    CM = CartesianIndices(M)
    Q = [0=>CI(1,1)]
    tot_risks = fill(typemax(Int), size(M))
    while true
        ith = last(findmin(first, Q))
        risk, here = popat!(Q, ith)
        tot_risks[here] <= risk && continue # not a better route
        tot_risks[here] = risk
        here == last(CM) && return last(tot_risks) # reached goal

        for ne in filter!(∈(CM), Ref(here) .+ adjs)
            push!(Q, risk + M[ne] => ne)
        end
    end
end
println("P1: ", main(M))
println("P2: ", main(M2))
