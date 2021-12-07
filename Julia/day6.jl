using LinearAlgebra
const ary = parse.(Int, split(readline("../input6"), ","))

function f(ary, days)
    counts = zeros(Int, 9)
    for a in ary
        counts[a+1] += 1
    end
    ker = Int[0 0 0 0 0 0 1 0 1
        I(8) zeros(8)]
    sum(counts' * ker^days)
end

println("P1: $(f(ary, 80)), P2: $(f(ary, 256))")
