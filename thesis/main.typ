#import "lib/mod.typ": *
#import "template.typ": *


#set raw(theme: "catppuccin.tmTheme")
#show figure.where(kind: raw): set block(breakable: true)

#show raw: it => {
  text(fill: theme.text, it)
}

#show raw.where(block: false): (it) => {
  set text(catppuccin.latte.text, font: "JetBrainsMono NFM", size: 1em)
  set box(fill: catppuccin.latte.base, radius: 3pt, stroke: none, inset: (x: 2pt), outset: (y: 2pt))
  box(
    it,
  )
}
#show footnote : it => {
  set text(accent)
  it
}

#show link: it => text(accent, it)


#set list(marker: {
  set line(stroke: (paint: accent))
  marker.arrow.single
})

#set enum(full: true)

#set grid(gutter: 0.5em)
#set image(width: 100%)

#show outline.entry.where() : it => {
  let t = it.body.fields().values().at(0)

  let size = 1em
  let color = accent.darken(40%)
  let weight = "medium"

  if it.level == 1 {
    v(1.5em)
    size = 1.2em
    color = accent
    weight = "black"
  }

  if type(t) == "array" {
    v(-3em)
    // h(it.element.level * 2em)
    let offset = if it.element.level == 1 {
      -1.25mm
    } else {
      0mm
    }
    link(
      it.element.location(),
      grid(
        column-gutter: 2pt,
        columns: ((it.element.level - 1) * 8mm + offset, auto, 1fr, auto),
        align: (center, left + bottom, center + bottom, right + bottom),
        [],
        text(color, size: size, weight: weight, [
          #box(

            width: 8mm,
            it.body.fields().values().at(0).at(0)
          )
          #it.body.fields().values().at(0).slice(2).join("")
        ]),
        //block(fill: color, height: 0.5pt, width: 100%),
        line(stroke: (paint: accent, thickness: 1pt, dash: "dotted"), length: 100%),
        text(color, size: 1em, weight: weight, it.page),
      )
    )
  } else {
    v(-3em)
    link(
      it.element.location(),
      grid(
        column-gutter: 2pt,
        columns: (0em, auto, 1fr, auto),
        align: (center, left + bottom, center + bottom, right + bottom),
        [],
        text(color, size: size, weight: weight, it.body),
        //block(fill: color, height: 0.5pt, width: 100%),
        line(stroke: (paint: accent, thickness: 1pt, dash: "dotted"), length: 100%),
        text(color, size: 1em, weight: weight, it.page),
      )
    )
  }
}

// OVERSKRIFTER
#show heading.where(numbering: "1.1") : it => [
  #v(0.75em)
  #block({
    box(width: 18mm, text(counter(heading).display(), weight: 600))
    text(it.body, weight: 600)
  })
  #v(1em)
]

#show heading.where(level: 1) : it => text(
  accent,
  size: 18pt,
)[
  #pagebreak(weak: true)
  #let subdued = accent.lighten(50%)
  #set text(font: "Roboto Mono")

  #grid(
    columns: (3fr, 1fr),
    align: (left + bottom, right + top),
    text(it.body, size: 24pt, weight: "light"),
    if it.numbering != none {
      text(subdued, weight: 200, size: 100pt)[#counter(heading).get().at(0)]
      v(0em)
    },
  )
  #v(-0.5em)
  // #hr
  #hline-with-gradient(cmap: (accent, subdued), height: 2pt)
  #v(1em)
]

#show heading.where(level: 4) : it => text(
  black,
  size: 14pt,
)[
  #v(0.25em)
  #block({
    box(width: 18mm, text(counter(heading).display(), weight: 600))
    text(it.body, weight: 600)
  })
  #v(0.15em)
]

#show: paper.with(
  paper-size: "a4",
  title: project-name,
  subtitle: "Master's Thesis in Computer Engineering",
  title-page: true,
  title-page-extra: align(center,
    std-block(
      width: 90%,
      std-block(
        radius: 0.5em,
        clip: true,
        inset: 0pt,
        pad(
          rest: -1mm,
          image("figures/img/pray4this.png")
        )
      )
    )
  ),
  title-page-footer: align(center)[
    #grid(
      columns: 2,
      row-gutter: 1em,
      align: (x, y) => (right, left).at(x),
      [*Supervisor:*],
      supervisors.lukas.name,
    )
    #v(1em, weak: true)
    lukas.esterle\@ece.au.dk
    #v(7mm, weak: true)
    #image("img/au-logo.svg", width: 30%)
  ],
  authors: authors,
  date: datetime.today().display("[day]-[month]-[year]"),

  print-toc: false,
  toc-depth: 2,
  accent: accent,
  postbody: context [
    #let resume-page = counter(page).at(<nomenclature>).first()

    #counter(page).update(resume-page + 2)
    #set page(numbering: "i")
    #pagebreak(weak: true)
    #start-appendix(show-toc: true)

    #show heading : it => text(
      black,
    )[
      #set par(justify: false)
      #v(0.25em)
      #block([
        #text(counter(heading).display()) #text(it.body, weight: 600)
      ])
      #v(0.15em)
    ]

    #show heading.where(level: 1) : it => text(
      accent,
      font: "Roboto Mono",
    )[
      #pagebreak(weak: true)
      #set par(justify: false)
      #v(0.25em)
      #block([
        #text(counter(heading).display()) #text(it.body, weight: 600)
      ])
      #v(0.15em)
    ]

    #set heading(numbering: "A.1:", supplement: "Appendix")
    #include "sections/appendix.typ"
  ]
)

#let acronyms = yaml("acronyms.yaml")

#let acrostiche-acronyms = merge(..acronyms.map(it => {
  let v = (it.definition,)
  if "plural" in it {
    v.push(it.plural)
  }

  (it.acronym: v)
}))

#init-acronyms(acrostiche-acronyms)

// This is important! Call it whenever your page is reconfigured.
// #if not release {
//   set-page-properties()
// }

// #if "release" in sys.inputs and sys.inputs.release == "true" {
//   set-margin-note-defaults(hidden: true)
// } else {
//   set-margin-note-defaults(hidden: false)
// }
// #show: word-count

// Pre-introduction
#set heading(numbering: none)
// #set page(numbering: "i")
#include "sections/0-predoc/preface.typ"
#include "sections/0-predoc/abstract.typ"
#include "sections/0-predoc/nomenclature.typ"
#include "sections/0-predoc/acronym-index.typ"

// Table of Contents
#pagebreak(weak: true)
#heading([Contents], level: 1, numbering: none, outlined: false)<contents-1>

#v(1em)
#toc-printer(target: heading.where().before(<contents-1>).and(heading.where(level: 1)))

#let main-numbering = "1.1"
#v(1em)
#toc-printer(target: heading.where(numbering: main-numbering))
#v(1em)
#toc-printer(target: heading.where(numbering: none).after(<contents-1>).before(<references>))
#v(1em)
#toc-printer(target: heading.where(numbering: none).after(<appendix>))

// #stats()

// Report
#set heading(numbering: main-numbering)
#set page(numbering: "1")
#counter(heading).update(0)
#counter(page).update(1)


#set page(
  header: context {
    let h1 = hydra(1)
    let h2 = hydra(2)
    if h1 == none {
      return none
    }

    let cm = (accent, theme.text.lighten(50%))
    let odd = calc.odd(here().page())

    if odd {
      cm = cm.rev()
    }
    set text(gradient.linear(..cm))

    let h = {
      let chapter-number = h1.fields().values().first().first()
      let chapter-title = h1.fields().values().first().slice(2)

      if type(chapter-title) == "array" {
        chapter-title = chapter-title.join("")
      }

      [Chapter #chapter-number: #chapter-title]

      // repr(h1.fields().values().first().slice(2))
      if h2 != none {
        h(0.5em)
        sym.bar.h
        h(0.5em)
        h2
      }
    }

    let date = {
      datetime.today().display("[day]-[month]-[year]")
    }
    let items = (h, date)
    if odd {
      items = items.rev()
    }

    grid(
      columns: (auto, 1fr),
      align: (left, right),
      ..items
    )
    hline-with-gradient(cmap: cm, height: 1pt)
  }
)

#include "sections/1-introduction/mod.typ"
#include "sections/2-background/mod.typ"
#include "sections/3-methodology/mod.typ"
#include "sections/4-results/mod.typ"
#include "sections/5-discussion/mod.typ"
#include "sections/6-conclusion/mod.typ"

#heading([References], level: 1, numbering: none)<references>
#bibliography(
  "./references.yaml",
  // style: "future-science",
  // style: "american-society-of-mechanical-engineers",
  style: "ieee",
  title: none,
)
