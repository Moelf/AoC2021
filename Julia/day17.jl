const AX, AY = eval(Meta.parse("($(match(r"x.*", replace(readchomp("../input17"), ".."=>":")).match))"))

let y_max = 0, res=0
    for vx=-AX.stop:AX.stop, vy=AY.start:-AY.start
        x=y=_ymax=0
        while true
            x,y,vx,vy = x+vx, y+vy, vx - sign(vx), vy-1
            _ymax = max(y, _ymax)
            if x ∈ AX && y ∈ AY
                y_max = max(y_max, _ymax)
                res+=1
                break
            end
            y <= AY.start && vy < 0 && break # impossible to come back
        end
    end
    println("P1: ", y_max)
    println("P2: ", res)
end
