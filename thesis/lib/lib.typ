
#import "blocks.typ": *
#import "note.typ"
#import "colors.typ"
#import "catppuccin.typ": *
// #import "template.typ": *
#import "@preview/codelst:2.0.2": sourcecode, codelst, sourcefile

#import "@preview/tablex:0.0.6": *
#import "@preview/drafting:0.2.0": *
#import "@preview/cetz:0.2.2": *
#import "@preview/hydra:0.4.0": hydra
// #import "@preview/glossarium:0.3.0": make-glossary, print-glossary, gls, glspl
// #show: make-glossary

#import "@preview/wordometer:0.1.4": word-count, total-words, total-characters
// #show: word-count.with(exclude: (heading.where(level: 1), strike, figure.caption, <no-wc>))

#import "@preview/acrostiche:0.5.0": init-acronyms, print-index, acr, acrpl
#import "@preview/oxifmt:0.2.1": strfmt

// #show: word-count

#let theme = catppuccin.latte
#let accent = theme.sapphire.darken(10%)
// #let colors = (
// )

#let rep(item, n) = range(n).map(_ => item)

#let hr = line(length: 100%)

#let project-name = "Pre-planning Intersection Traversal for Autonomous Vehicles"

#let supervisors = (
  lukas: (
    name: "Lukas Esterle",
    email: "lukas.esterle@ece.au.dk",
  )
)

#let authors = ((
  name: "Ian Dahl Oliver",
  email: "ian.oliver@post.au.dk",
  auid: "au672461",
)
,).map(author => {
  author + (
    department: [Department of Electrical and Computer Engineering],
    organization: [Aarhus University],
    location: [Aarhus, Denmark],
  )
})

#let a = (
  ian: authors.at(0).name,
)

#let sourcecode = sourcecode.with(frame: std-block, numbers-style: (lno) => move(dy: 1pt, text(
  font: "JetBrainsMono NFM",
  size: 0.75em,
  catppuccin.latte.overlay0,
  lno,
)))

#let sourcefile = sourcefile.with(frame: std-block, numbers-style: (lno) => move(dy: 1pt, text(
  font: "JetBrainsMono NFM",
  size: 0.75em,
  catppuccin.latte.overlay0,
  lno,
)))

#let sourcecode-reference(content, caption: none) = {
  return figure(
    content,
    kind: "code",
    supplement: [Listing],
    caption: caption
  )
}

// #let code-snippet(content) = {
//   figure(
//     content,
//     kind: "code",
//     supplement: [Code Snippet]
//   )
// }

#let snippet = sourcecode.with(numbering: none, frame: std-block.with(inset: 2pt, radius: 3pt))

#let tablex = tablex.with(stroke: 0.25pt, auto-lines: false, inset: 0.5em)

#let rule = hlinex(stroke: 0.25pt)
#let thick-rule = hlinex(stroke: 0.75pt)
#let double-rule(cols) = (hlinex(), colspanx(cols)[#v(-0.75em)], hlinex(),)

#let boxed(content, color: accent.lighten(0%), fill: none, weight: 400) = {
  if (fill == none) {
    fill = color.transparentize(80%)
  }
  box(
    fill: fill,
    radius: 3pt,
    inset: (x: 2pt),
    outset: (y: 2pt),
    text(color, weight: weight, content),
  )
}

// #let list-box(title, content) = {
//   gridx(
//     columns: (1em, 1fr),
//     column-gap: 1pt,
//     [], [#boxed(title) #content],
//   )
//   // v(-3.25em)
// }

#let no-indent(content) = {
  set par(first-line-indent: 0pt)
  content
}

#let remark(txt, color: color.red, prefix: "") = {
  if not "release" in sys.inputs {
    boxed(color: color.lighten(0%), fill: color.lighten(80%), [*#prefix* #txt])
  }

  state("remark" + prefix, 0).update(s => s + 1)
}

#let todo = remark.with(color: theme.maroon, prefix: "Todo: ")
#let jens = remark.with(color: theme.teal, prefix: "Jens: ")
#let yens = remark.with(color: theme.pink, prefix: "Yens: ")
#let kristoffer = remark.with(color: theme.green, prefix: "Kristoffer: ")
#let k = kristoffer
#let jonas = remark.with(color: theme.mauve, prefix: "Jonas: ")
#let att(content) = note.a[] + text(theme.peach, content)

#let boxed-enum(
  prefix: "",
  suffix: "",
  delim: ".",
  color: accent.lighten(0%),
  fill: accent.lighten(80%),
  ..numbers,
) = boxed(
  color: color,
  fill: fill,
  prefix + numbers.pos().map(str).join(delim) + delim + suffix,
  weight: 900,
)

#let box-enum(
  prefix: "",
  suffix: "",
  delim: ".",
  color: accent.lighten(0%),
  ..numbers,
) = boxed(
  color: color,
  fill: color.lighten(80%),
  prefix + numbers.pos().map(str).join(delim) + suffix,
  weight: 900,
)

#let hyp-counter = counter("hypothesis")
#let h-enum(
  ..numbers,
) = {
  hyp-counter.step()
  box-enum(prefix: "H-", delim: ".", color: theme.lavender, ..numbers)
}

#let req-enum(prefix: "R-", color: accent, ..numbers) = boxed(
  color: color,
  fill: color.lighten(80%),
  text(weight: 900, prefix + numbers.pos().map(n => {
    if n < 10 {
      str(n)
    } else {
      str(n)
    }
  }).join(".")),
)

#let vdmpp(text) = raw(text, block: false, lang: "vdmpp")

#let requirement-heading(content) = heading(
  level: 2,
  supplement: "Requirement",
  numbering: none,
  outlined: false,
  content,
)
#let req(content) = {
  boxed(weight: 900)[R-#content]
  h(0.5em)
}

#let abstraction-counter = counter("abstraction-counter")

#let abstraction-id = {
  abstraction-counter.step()
  boxed("A" + context abstraction-counter.display(), weight: 900)
}

#let cy(content) = text(catppuccin.latte.yellow, content)
#let cr(content) = text(catppuccin.latte.maroon, content)
#let cg(content) = text(catppuccin.latte.green, content)
#let cb(content) = text(catppuccin.latte.lavender, content)
#let cp(content) = text(catppuccin.latte.mauve, content)

#let ra = sym.arrow.r
#let la = sym.arrow.l

#let swatch(color, content: none, s: 6pt) = {
  if content != none {
    content
    // h(0.1em)
  }
  h(1pt, weak: true)
  box(
    height: s,
    width: s,
    fill: color,
    // stroke: 1pt + color,
    radius: s / 2,
    baseline: (s - 0.5em) / 2,
  )
}

#let sy = swatch(catppuccin.latte.yellow)
#let sr = swatch(catppuccin.latte.maroon)
#let sg = swatch(catppuccin.latte.green)
#let so = swatch(catppuccin.latte.peach)
#let sb = swatch(catppuccin.latte.blue)
#let sp = swatch(catppuccin.latte.mauve)
#let sl = swatch(catppuccin.latte.lavender)
#let sgr = swatch(catppuccin.latte.surface0)
#let sgr2 = swatch(catppuccin.latte.surface2)
#let sgr3 = swatch(catppuccin.latte.overlay2)
#let st = swatch(catppuccin.latte.text)
#let stl = swatch(catppuccin.latte.teal)

#let nameref(label, name, supplement: none) = {
  show link : it => text(accent, it)
  link(label, [#ref(label, supplement: supplement). #name])
}

#let numref(label) = ref(label, supplement: none)

#let scen(content) = boxed(color: catppuccin.latte.yellow, content)

#let toc-printer(target: none, depth: 2) = {
  set par(first-line-indent: 0em)
  outline(
    indent: 2em,
    fill: grid(
      columns: 1,
      block(fill: black, height: 0.5pt, width: 100%),
    ),
    depth: depth,
    target: target,
    title: none,
  )
}

#let hline-with-gradient(cmap: color.map.inferno, height: 2pt) = rect(width: 100%, height: height, fill: gradient.linear(..cmap))

#let merge(..dicts) = {
  dicts.pos().fold((:), (acc, dict) => {
    for (k, v) in dict {
      acc.insert(k, v)
    }
    acc
  })
}

#let important-datetimes = (project: (
  start: datetime(day: 27, month: 01, year: 2025),
  end: datetime(day: 05, month: 06, year: 2024),
))

#let plural(word, n) = if n <= 1 {
  word
} else {
  word + "s"
}

#let as-string(any) = {
    if type(any) == "string" {
        any
    } else if type(any) == "content" {
        let repr_any = repr(any)
        repr_any.slice(1, repr_any.len() - 1) // remove square brackets
    } else {
        str(any)
    }
}

#let plural-alt(s) = {
    let s = as-string(s)
    if s.ends-with("s") {
        // plural
        s + "es"
    } else {
        // singular
        s + "s"
    }
}

#let possessive(s) = {
    let s = as-string(s)
    if s.ends-with("s") {
        // plural
        s + "s"
    } else {
        // singular
        s + "'s"
    }
}

// Format a string | content as Title Case
#let titlecase(text) = {
    let string = if type(text) == "string" {
        text
    } else if type(text) == "content" {
        repr(text).slice(1,-1) // remove square brackets
    } else {
        panic("Invalid type for text. Valid types are 'string' | 'content'")
    }

    string.split(" ").map(word => {
        let chars = word.split("")
        (
            upper(chars.at(1)),
            ..chars.slice(2, -1)
        ).join("") // join into a string again
    }).join(" ") // join into a sentence again
}

#let repo(org: none, repo: none) = {
    if (repo == none) {
        panic("Name is required for repo")
    }
    if (org == none) {
        raw(repo)
    }
    raw((org, repo).join("/"), block: false)
}

#let release = "release" in sys.inputs and sys.inputs.release == "true"

#let stats() = {
  if release {
    return
  }
  locate(loc => {
    let words = state("total-words").final(loc)
    let chars = state("total-characters").final(loc) + words * 0.8
    let normal-pages = chars / 2400

    let total-pages = 80
    let people = 2

    let total-days = important-datetimes.project.end - important-datetimes.project.start
    let days-left = important-datetimes.project.end - datetime.today()

    // let pages-person-day = (total-pages - normal-pages) / (people * days-left).days()

    set text(size: 10pt, font: "JetBrainsMono NFM")
    set par(first-line-indent: 0em)
    set align(center)

    tablex(
      columns: (auto, auto),
      align: (left, right),
      [*words*], [#words],
      [*characters*], [#calc.round(chars, digits: 0)],
      [*normal pages*], [#calc.round(normal-pages, digits: 2)],
      rule,
      [*goal pages*], [#total-pages],
      [*goal characters*], [#(total-pages * 2400)],
      // [*pp./person/day*], [#calc.round(pages-person-day, digits: 2)],
      [*days left*], [#days-left.days()],
    )

    let colors = (
      complete: catppuccin.latte.lavender,
      incomplete: catppuccin.latte.maroon,
    )
    let progress = normal-pages / total-pages * 100%
    let progress-left = 100% - progress

    let days-gone = total-days.days() - days-left.days()
    let days-left-percent = days-left.days() / total-days.days() * 100%

    grid(
      column-gutter: 0pt,
      columns: (1fr, auto),
      row-gutter: 5pt,
      text(colors.complete, [#repr(progress) (#calc.round(normal-pages, digits: 2) pages)]), text(colors.incomplete, [#repr(progress-left) (#calc.round(total-pages - normal-pages, digits: 2) pages)]),
    )
    v(-0.75em)
    grid(
      column-gutter: 0pt,
      columns: (progress, auto),
      row-gutter: 5pt,

      box(height: 1em, width: 100%, fill: colors.complete),
      box(height: 1em, width: 100%, fill: colors.incomplete),
    )

    let percent-to-indhent = 100% - progress - days-left-percent
    grid(
      column-gutter: 0pt,
      columns: (progress, 1fr, days-left-percent),
      row-gutter: 5pt,
      [], text(colors.incomplete, [#repr(percent-to-indhent)]), [],
      box(height: 1em, width: 100%, fill: theme.overlay0), box(height: 1em, width: 100%, fill: colors.incomplete), box(height: 1em, width: 100%, fill: theme.overlay0),
    )


    v(0.25em)
    grid(
      column-gutter: 0pt,
      columns: (1fr, auto),
      row-gutter: 5pt,
      text(colors.incomplete, [#repr(100% - days-left-percent) (#days-gone days)]),
      text(colors.complete, [#repr(days-left-percent) (#days-left.days() days)]),
    )
    v(-0.75em)
    grid(
      column-gutter: 0pt,
      columns: (1fr, days-left-percent),
      row-gutter: 5pt,
      box(height: 1em, width: 100%, fill: colors.incomplete),
      box(height: 1em, width: 100%, fill: colors.complete),
    )

    let t = state("remark" + "Todo: ").final(loc)
    let j = state("remark" + "Jens: ").final(loc)
    let k = state("remark" + "Kristoffer: ").final(loc)
    let jo = state("remark" + "Jonas: ").final(loc)
    let total = t + j + k + jo

    let columns = ()
    let texts = ()
    let boxes = ()

    if t != none {
      columns.push(t)
      texts.push(text(theme.maroon, [#t todo]))
      boxes.push(box(height: 1em, width: 100%, fill: theme.maroon))
    }
    if j != none {
      columns.push(j)
      texts.push(text(theme.teal, [#j Jens]))
      boxes.push(box(height: 1em, width: 100%, fill: theme.teal))
    }
    if k != none {
      columns.push(k)
      texts.push(text(theme.green, [#k Kristoffer]))
      boxes.push(box(height: 1em, width: 100%, fill: theme.green))
    }
    if jo != none {
      columns.push(jo)
      texts.push(text(theme.mauve, [#jo Jonas]))
      boxes.push(box(height: 1em, width: 100%, fill: theme.mauve))
    }

    v(1em)

    grid(
      column-gutter: 0pt,
      columns: columns.map(v => v / total * 100%),
      row-gutter: 5pt,
      ..texts,
      ..boxes,
    )

    align(center, [#total *#plural("remark", total)*])
  })
}

#let print-index(level: 1, outlined: false, sorted: "", title: "Acronyms Index", delimiter:":") = {
  pagebreak(weak: true)
  set page(columns: 2)
  // assert on input values to avoid cryptic error messages
  assert(sorted in ("","up","down"), message:"Sorted must be a string either \"\", \"up\" or \"down\"")

  if title != ""{
    heading(level: level, outlined: outlined)[#title]
  }

  state("acronyms",none).display(acronyms=>{

    // Build acronym list
    let acr-list = acronyms.keys()

    // order list depending on the sorted argument
    if sorted!="down"{
      acr-list = acr-list.sorted()
    }else{
      acr-list = acr-list.sorted().rev()
    }

    let to-content-array() = {
      let arr = ()
      for acr in acr-list {
        let acr-long = acronyms.at(acr)
        let acr-long = if type(acr-long) == array {
          acr-long.at(0)
        } else {acr-long}
        // ([*#acr#delimiter*], [#acr-long\ ])
        arr.push([*#acr#delimiter*])
        arr.push([#acr-long\ ])
      }
      arr
    }

    tablex(
      columns: (auto, 10fr),
      column-gutter: 1em,
      row-gutter: 0.75em,
      align: (right, left),
      inset: 0pt,
      // (), vlinex(), (),
      ..to-content-array()
    )
  })
}

#let node(color, content, rounding: 50%, size: 4mm) = {
  let width = if (repr(content).len() - 2) > 1 { 0pt } else { 1.5pt };
  h(5pt)
  // block(
    box(
      fill: color.lighten(90%),
      stroke: 1pt + color,
      outset: (x: size / 2, y: size / 2),
      inset: (y: -size / 4),
      radius: rounding,
      baseline: -size / 4,
      height: 0pt,
      width: 0pt,
      align(
        center,
        text(
          catppuccin.latte.text,
          size: 0.5em,
          weight: "bold",
          font: "JetBrains Mono",
          content
        )
      )
    )
  // )
  h(5pt)
}

#let variable(color, content) = {
  node(color, content)
}

#let factor(color, content) = {
  node(color, content, rounding: 5%, size: 1.25mm)
}

#let listing-counter = counter("listing")

#let listing(
  content,
  line-numbering: auto,
  caption: none,
) = {
  let supplement = [Listing]

  let sourcecode = if line-numbering == none {
    sourcecode.with(numbers-style: (lno) => none)
  } else {
    sourcecode
  }

  return figure(
    {
      sourcecode(content)
    },
    caption: caption,
    kind: "listing",
    supplement: supplement,
  )
}

#let example-counter = counter("example")
#let example-box(number) = boxed(color: accent)[*Example #number:*]

#let example(
  body,
  caption: none,
) = {
  let supplement = [Example]
  let n = context example-counter.get().at(0)
  let title_prefix = text(weight: "bold", "Example " + n + if caption != none {": "} else {""})

  return figure(
    {
      // set text(size: 0.8em)
      example-counter.step()
      blocked(
        title: text(accent, title_prefix) + text(weight: "regular", caption),
        body + [#metadata("example") <meta:excounter>],
        color: theme.lavender.lighten(90%),
      )
    },
    kind: "example",
    supplement: [Example],
  )
}

#let algorithm-counter = counter("algorithm")

#let algorithm(
    content,
    caption: none,
) = {
    let supplement = [Algorithm]
    let n = context algorithm-counter.get().at(0)
    let title_prefix = text(weight: "bold", "Algorithm " + n + if caption != none {": "} else {""})
    let ind = 1em

    return figure(
      {
        // set text(size: 0.8em)
        set par(first-line-indent: ind, hanging-indent: ind)
        algorithm-counter.step()

        blocked(
          title: text(theme.sapphire.lighten(20%), title_prefix) + text(weight: "regular", caption),
          h(ind) + content,
        color: theme.sapphire.lighten(90%),
          // content + linebreak() +
          // repr(content.fields())
        )
      },
      numbering: "1.",
      kind: "algorithm",
      supplement: supplement,
    )
}


#let H(n) = [Hypothesis #boxed(color: theme.lavender)[*H-#n*]]
#let RQ(n) = [Research Question #boxed(color: theme.teal)[*RQ-#n*]]
#let O(n) = [Objective #boxed(color: theme.lavender)[*O-#n*]]

#let scen = (
  circle: "Circle",
  environment-obstacles: "Environment Obstacles",
  varying-network-connectivity: "Varying Network Connectivity",
  // clear-circle: "Clear Circle",
  junction: "Junction",
  communications-failure: "Communications Failure",
  solo-gp: "Solo Global Planning",
  collaborative-gp: "Collaborative Global Planning",
  iteration-amount: "Iteration Amount",
  iteration-schedules: "Iteration Schedules",
).pairs().map(
  it => {
    let key = it.at(0)
    let value = it.at(1)

    (
      key,
      (
        s: text(style: "italic", weight: 900, value),
        n: value
      )
    )
  }
).fold((:), (acc, it) => {
  acc.insert(it.at(0), it.at(1))
  acc
})

#let study = (
  heading: it => {
    set par(first-line-indent: 0em)
    v(0.25em)
    block(text(size: 14pt, weight: 900, it))
  },
  H-1: (
    box: boxed(text(weight: 900, "H-1")),
    prefix: [_*Contribution 1*_],
    name: [_*Simulation Framework*_],
    full: (
      s: [_*Contribution 1 - Simulation Framework*_],
      n: [Contribution 1 - Simulation Framework],
    )
  ),
  H-2: (
    box: boxed(text(weight: 900, "H-2")),
    prefix: [_*Contribution 2*_],
    name: [_*Algorithm Enhancements*_],
    full: (
      s: [_*Contribution 2: Algorithm Enhancements*_],
      n: [Contribution 2: Algorithm Enhancements],
    )
  ),
  H-3: (
    box: boxed(text(weight: 900, "H-3")),
    prefix: [_*Contribution 3*_],
    name: [_*Global Planning Layer*_],
    full: (
      s: [_*Contribution 3: Global Planning Layer*_],
      n: [Contribution 3: Global Planning Layer],
    )
  ),
  H-4: (
    box: boxed(text(weight: 900, "H-4")),
    prefix: [_*Contribution 4*_],
    name: [_*GBP Path Tracking*_],
    full: (
      s: [_*Contribution 4: GBP Path Tracking*_],
      n: [Contribution 4: GBP Path Tracking],
    )
  ),
)

#let step = (
  s1: boxed(color: colors.variable)[*Step 1*],
  s2: boxed(color: colors.variable)[*Step 2*],
  s3: boxed(color: colors.factor)[*Step 3*],
  s4: boxed(color: colors.factor)[*Step 4*],
)

#let iteration = (
  // factor: boxed(color: colors.factor)[*Factor Iteration*],
  // variable: boxed(color: colors.variable)[*Variable Iteration*],
  factor: text(colors.factor, "Factor Iteration"),
  variable: text(colors.variable, "Variable Iteration"),
)

#let cost = (
  cheap: " " + sg + " " + text(theme.green, style: "italic", "Cheap"),
  expensive: " " + sr + " " + text(theme.maroon, style: "italic", "Expensive"),
)

#let gaussian = (
  moments: [_Moments Form_],
  canonical: [_Canonical Form_],
)

#let inference = (
  MAP: [_#acr("MAP") inference_],
  marginal: [_marginal inference_],
)

#let factor = (
  lp: [_linearization point_],
)


#let jacobian = $upright(bold(J))$
#let m = (
  Lambda: $#text(theme.mauve, $Lambda$)$,
  eta: $#text(theme.mauve, $eta$)$,
  mu: $#text(theme.mauve, $mu$)$,
  Sigma: $#text(theme.mauve, $Sigma$)$,
  X: $#text(theme.mauve, $X$)$,
  J: jacobian,
  SA: $#text(theme.teal, $Sigma_A$)$,
  SB: $#text(theme.mauve, $Sigma_B$)$,
  // #let m = (
  Xb: $bold(upright(X))$,
  D: $bold(upright(D))$,
  x: $bold(upright(x))$,
  P: $bold(upright(P))$,
  p: $bold(upright(p))$,
  proj: $bold(upright("proj"))$,
  d: $bold(upright(d))$,
  l: $bold(upright(l))$,
  where: h(0.5em) + line(start: (0em, -0.1em), end: (0em, 1.2em), stroke: 0.25pt) + h(0.5em)
// )
)

#let number-word(num) = {
  let num = str(num)
}


#let fsig(content) = text(font: "JetBrainsMono NF", size: 0.85em, content)
#let algeq(content) = {
  show regex("(SampleRandomPoint|NearestNeighbor|Steer|CollisionFree|WithinGoalTolerance|MinCostConnection|Rewire|Sample|Nearest|ObstacleFree|neighborhood|Cost|Line|Parent)"): set text(theme.mauve, font: "JetBrainsMono NFM", size: 0.85em)
  content
}

#let transpose(matrix) = {
  assert(type(matrix) == array)
  let ncols = calc.max(..matrix.map(row => row.len()))
  let nrows = matrix.len()

  assert(matrix.map(row => row.len()).all(len => len == ncols))

  for col in range(ncols) {
    (matrix.map(row => row.at(col)),)
  }
}

// https://github.com/AU-Master-Thesis/gbp-rs/blob/main/scripts/ldj.py
#let github(owner, repo, branch: "main", path: none, content: none) = {
  let url = "https://github.com/" + owner + "/" + repo
  if path != none {
    url = url + "/blob/" + branch + "/" + path
  }
  let name = if path == none {
    raw(repo)
  } else {
    raw(path)
  }

  link(url, name)
}

#let gbp-rs(content: none) = {
  let name = if content == none {
    "GitHub repository"
  } else {
    content
  }
  show regex("."): set text(accent)
  link("https://github.com/AU-Master-Thesis/gbp-rs", name)
}

#let source-link(dest, file-path) = {
  show regex("."): set text(accent)
  link(dest, raw(block: false, file-path))
}

#let panel = (
  bindings: text(theme.green, [*Keybindings Panel*]),
  viewport: text(theme.lavender, [*Viewport*]),
  settings: text(theme.maroon, [*Settings Panel*]),
  metrics: text(theme.peach, [*Metrics Panel*]),
)

#let z-stack(..items) = {
  grid(
    columns: items.pos().len() * (1fr,),
    column-gutter: -100%,
    rows: 1,
    ..items
  )
}

#let gradient-box(..cmap, width: 6em) = box(inset: (x: 2pt), outset: (y: 2pt), radius: 3pt, height: 0.5em, width: width, fill: gradient.linear(..cmap))


#let interleave(..arrays) = {
  let arrays = arrays.pos()
  assert(arrays.map(it => it.len()).all(len => len == arrays.at(0).len()))

  for i in range(arrays.at(0).len()) {
    for array in arrays {
      (array.at(i),)
    }
  }
}

// #let foo(dict) = {
//   for (k, list) in dict {
//   [#k]
//   for
//     (k, v)
//   }
// }


#let legend(handles, direction: ltr, fill: white.transparentize(25%)) = {
  std-block(
    width: auto,
    // fill: theme.base.transparentize(100%),
    fill: fill,
    // stroke:
    {
      set align(left)
      stack(
        dir: direction,
        ..handles.map(handle => {
          if direction == rtl {
            h(handle.space)
          } else if direction == btt {
            v(handle.space / 2)
          }
          let patch = if "patch" in handle {
            handle.patch
          } else {
            box(height: 0.8em, width: 0.8em, radius: 100%, fill: handle.color.lighten(handle.alpha), stroke: handle.color.lighten(handle.alpha), inset: (x: 2pt), baseline: 1pt)
          }
          box(
            width: 1em,
            // baseline: 0.15em,
            align(
              center,
              patch
            )
          )
          h(0.5em)
          text(handle.color, size: 0.8em, handle.label)
          if direction == ltr {
            h(handle.space)
          } else if direction == ttb {
            v(handle.space / 2)
          }
        })
      )
    }
  )
}


#let lm3-th13 = (
  s: text(theme.green, $l_m=3,t_(K-1)=13.33s$),
  n: $l_m=3,t_(K-1)=13.33s$
)
// #let lm1-th13 = text(theme.yellow, $l_m=1,t_(K-1)=13.33s$)
// #let lm3-th5 = text(theme.lavender, $l_m=3,t_(K-1)=5s$)
#let lm1-th13 = (
  s: text(theme.teal, $l_m=1,t_(K-1)=13.33s$),
  n: $l_m=1,t_(K-1)=13.33s$
)
#let lm3-th5 = (
  s: text(theme.lavender, $l_m=3,t_(K-1)=5s$),
  n: $l_m=3,t_(K-1)=5s$
)

#let inline-line(
  length: 1em,
  stroke: (paint: theme.lavender, thickness: 2pt, cap: "round")
) = box(
  line(
    length: length,
    stroke: stroke
  ),
  baseline: -0.25em
)

#let solo-gp-mean-decrease = (1 - 0.74 / 1.03) * 100
#let collaborative-gp-mean-decrease = (1 - 0.86 / 1.00) * 100

#let crop(
  left: 0pt,
  top: 0pt,
  right: 0pt,
  bottom: 0pt,
  x: 0pt,
  y: 0pt,
  rest: 0pt,
  body,
) = block(
  clip: true,
  pad(
    left: left,
    top: top,
    right: right,
    bottom: bottom,
    x: x,
    y: y,
    rest: rest,
    body,
  )
)

#let etal = [_et al._]


#let configs = (
  config: source-link("https://github.com/AU-Master-Thesis/gbp-rs/blob/main/config/simulations/Environment%20Obstacles%20Experiment/config.toml", "config.toml"),
  environment: source-link("https://github.com/AU-Master-Thesis/gbp-rs/blob/main/config/simulations/Environment%20Obstacles%20Experiment/environment.yaml", "environment.yaml"),
  formation: source-link("https://github.com/AU-Master-Thesis/gbp-rs/blob/main/config/simulations/Environment%20Obstacles%20Experiment/formation.yaml", "formation.yaml")
)
