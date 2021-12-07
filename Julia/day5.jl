using LinearAlgebra
CI = CartesianIndex
_sign(x1,x2) = x1>=x2 ? -1 : 1

lines = split.(readlines("../input5"), r",| -> ")
coords = map(lines) do line
    c = parse.(Int, line)
    step = CI(_sign(c[1], c[3]), _sign(c[2], c[4]))
    CI(c[1], c[2]):step:CI(c[3], c[4])
end
# part 1
M = zeros(Int, 1000, 1000)
for c in coords
    any(==(1), size(c)) && (M[c] .+= 1)
end
println("P1: ", sum(>=(2), M))
# part 2
for c in coords
    any(==(1), size(c)) || (M[diag(c)] .+= 1)
end
println("P2: ", sum(>=(2), M))
