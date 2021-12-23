function explode(s)
    count = 0
    target = 0:-1
    for (i, c) in enumerate(s)
        count += c == '['
        count -= c == ']'
        if count == 5
            target = i:findnext(']', s, i)
            break
        end
    end
    if !isempty(target)
        _s = s[target]
        tar_l, tar_r = parse.(Int, x.match for x in eachmatch(r"\d+", _s))
        s = replace(s, _s=>"   0"; count=1)
        l = findnext(r"\d+", s, max(target.start-5, 1))
        if !isnothing(l) && l.stop+1 != target.stop
            l_res = parse(Int, s[l]) + tar_l
            s = s[begin:l.start-1] * string(l_res) * s[l.stop+1:end]
        end
        r = findnext(r"\d+", s, target.stop+1)
        if !isnothing(r)
            r_res = parse(Int, s[r]) + tar_r
            s = s[begin:r.start-1] * string(r_res) * s[r.stop+1:end]
        end
    end
    return replace(s, " "=>"")
end

function spli(s)
    idxs = findall(r"\d+", s)
    iid = findfirst(x-> parse(Int, s[x]) >= 10, idxs)
    isnothing(iid) && return s
    num_str = s[idxs[iid]]
    num = parse(Int, num_str)
    l,r = fld(num, 2), cld(num, 2)
    replace(s, num_str => "[$l,$r]"; count=1)
end

function ⊗(s1, s2)
    _s = s = "[$s1,$s2]"
    while true
        s = explode(s)
        if s != _s
            _s = s
            continue
        end
        s = spli(s)
        if s != _s
            _s = s
            continue
        end
        break
    end
    return s
end
