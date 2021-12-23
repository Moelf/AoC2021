using Base.Iterators: Stateful, take, cycle
p1, p2 = parse.(Int, last.(eachline("../input21")))

function game(p1,p2)
    det_die = Stateful(cycle(1:100))
    s1 = s2 = 0
    while true
        p1 = mod1(p1 + sum(take(det_die, 3)), 10)
        s1 += p1
        s1 >= 1000 && return det_die.taken * s2

        p2 = mod1(p2 + sum(take(det_die, 3)), 10)
        s2 += p2
        s2 >= 1000 && return det_die.taken * s1
    end
end

println("P1: ", game(p1,p2))
