const pairs = [=>(split(x, '-')...) for x in eachline("../input12")]

function islegal(name, path, twice)
    name == "start" && return false
    if name ∉ path || all(isuppercase, name)
        return true
    elseif twice[]
        twice[] = false
        return true
    end
    return false
end

function explore(path=["start"]; res=[0], twice=[false])
    here = last(path)
    if here == "end" 
        res[] += 1
        return res[]
    end
    for p in pairs
        t1 = copy(twice); t2 = copy(twice)
        p[1] == here && islegal(p[2], path, t1) && explore([path; p[2]]; res, twice=t1)
        p[2] == here && islegal(p[1], path, t2) && explore([path; p[1]]; res, twice=t2)
    end
    res[]
end

println("P1 :", explore())
println("P2 :", explore(twice=[true]))
