#let as-set(xs) = eval("${ " + xs.map(str).join(", ") + " }$")
#let as-list(xs) = eval("$[ " + xs.map(str).join(", ") + " ]$")

#let overdot(x) = math.accent(x, sym.dot)
