collapse_temp = """
<details>
<summary>SUMMARY</summary>

CONTENT
</details>
"""

open("./README.md", "w") do out
    for day in 1:30
        println(out, "## Day $day")
        for (a,b,c) in (("Julia", "julia", "jl"), ("OCaml", "ocaml", "ml"), 
                ("Python", "python", "py"), ("Cpp", "cpp", "cpp"))
                try
                folder = a == "Cpp" ? "Cpp/src" : a
                file = a == "Cpp" ? "day$day/main.$c" : "day$day.$c"
                code = read(joinpath(folder, file), String)
                body = replace(collapse_temp, "SUMMARY" => a, "CONTENT" => 
"""\
```$b
$code
```""")
                println(out, body)
                catch
                end
        end
    end
end
