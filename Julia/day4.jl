inputs = split(strip(read("../input4", String), '\n'), "\n\n")
draws = parse.(Int, split(inputs[1], ","))
boards = map(inputs[2:end]) do board
    parse.(Int, mapreduce(split, hcat, split(board, "\n")))
end
p = fill(-1,5)
wincon(m) = any(==(p), eachrow(m)) || any(==(p), eachcol(m))

# part 1 & 2
done = Set{Int}()
res = Int[]
for num in draws, (i, b) in enumerate(boards)
    replace!(b, num => -1)
    if i∉done && wincon(b)
        push!(done, i)
        push!(res, num * sum(filter(>(0), b)))
    end
end
println(res[[begin, end]])
