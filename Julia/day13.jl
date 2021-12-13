using SparseArrays

raw_p, raw_i = strip.(split(read("../input13", String), "\n\n"))
points = [parse.(Int, (y,x)) .+ 1 for (x,y) in split.(split(raw_p, "\n"), ",")]
instructions = split(raw_i, "\n")

let M = sparse(first.(points), last.(points), true)
    for (i, line) in enumerate(instructions)
        M = if contains(line, "y=")
            M[1:end÷2, :] .|| M[end:-1:end÷2+2, :]
        else
            M[:, 1:end÷2] .|| M[:, end:-1:end÷2+2]
        end
        i == 1 && println("P1: ", sum(!iszero, M))
    end
    display(M)
end
