function sol(N=50)
    M = falses(2N+1, 2N+1, 2N+1)
    for l in eachline("../input22")
        mat = eachmatch(r"(-?\d+)\.\.(-?\d+)", l)
        coords = mapreduce(x->parse.(Int, x.captures), vcat, mat) .+ (N+1)
        any(c -> (c<0 || c>2N), coords) && continue
        x1,x2,y1,y2,z1,z2  = coords
        region = CartesianIndices((x1:x2, y1:y2, z1:z2))
        val = startswith(l, "on") ? 1 : 0
        M[region] .= val
    end
    sum(M)
end

println("P1: ", sol())
