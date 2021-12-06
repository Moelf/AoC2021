const ary = parse.(Int, split(readline("./input6"), ","))

function f(ary, days)
    counts = zeros(Int, 10)
    for a in ary
        counts[a+2] += 1
    end
    for _ = 1:days
        counts = circshift(counts, -1)
        N_mature = first(counts)
        counts[end] = N_mature
        counts[6+2] += N_mature
    end
    sum(counts[2:end])
end

println("P1: $(f(ary, 80)), P2: $(f(ary, 256))")
