using SparseArrays
const CI = CartesianIndex
const algo = parse.(Int, collect(replace(readlines("../input20")[1], '.'=>0, '#'=>1)));
const raw_mat = Matrix{Int64}(mapreduce(l->permutedims(replace(collect(l), '.'=>0, '#'=>1)), vcat, readlines("../input20")[3:end]))

function conv(M, ci)
    window = ci-CI(1,1):ci+CI(1,1)
    idx = evalpoly(2, vec(M[window]') |> reverse) + 1
    algo[idx]
end

function main(N=2)
    M1 = raw_mat
    _s = size(M1, 1) + 2*N + 4
    M2 = zeros(Int, _s, _s)
    M2[3+N:end-N-2, 3+N:end-N-2] .= M1
    for _ = 1:N
        M1 = copy(M2)
        for c in CartesianIndices((_s-2, _s-2)) .+ CI(1,1)
            M2[c] = conv(M1, c)
        end
    end
    count(>(0), M2[N+1:end-N, N+1:end-N])
end

println("P1: ", main())
