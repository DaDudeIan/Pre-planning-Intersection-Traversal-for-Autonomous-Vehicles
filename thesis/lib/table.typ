#import "blocks.typ": *
#import "@preview/funarray:0.4.0": *
#import "marker.typ"

#let term-table(colors: (catppuccin.latte.lavender, ), ..rows) = {
  // insert a marker.arrow.single as a third column between the terms and their definitions

  let colors = cycle(colors, rows.pos().len())

  let rows = chunks(rows.pos(), 2).enumerate().map(el => {
    let index = el.at(0)
    let term-and-def = el.at(1)


    let term = term-and-def.at(0)
    let definition = term-and-def.at(1)
    (term, {
      set line(stroke: (paint: colors.at(index)))
      marker.arrow.single
    }, definition)
  }).flatten()

  // repr(rows)
  v(-0.5em)
  table(
    columns: (auto, auto, 1fr),
    stroke: none,
    row-gutter: 0.25em,
    ..rows
  )
  v(-0.5em)
}

#let tablec(
  title: none,
  columns: none,
  header: auto,
  alignment: auto,
  stroke: none,
  header-color: (
    fill: catppuccin.latte.lavender,
    text: white
  ),
  even-color: catppuccin.latte.mantle,
  odd-color: catppuccin.latte.base,
  fill: auto,
  ..content
) = {
  let column-amount = if type(columns) == int {
    columns
  } else if type(columns) == array {
    columns.len()
  } else {
    1
  }

  let header-rows = (-1, )

  if header != auto {
    let cells-in-header = header.children.map(it => {
      let internal-len = 0

      if not (it.has("colspan") or it.has("rowspan")) {
        internal-len = 1
      } else{
        if it.has("rowspan") {
          internal-len += it.rowspan
        }
        if it.has("colspan") {
          internal-len += it.colspan
        }
      }
      internal-len
    }).fold(0, (acc, it) => acc + it)

    header-rows = range(int(cells-in-header / column-amount))
  }

  show table.cell : it => {
    if it.y in header-rows {
      set text(header-color.text)
      strong(it)
    } else if calc.even(it.y) {
      set text(catppuccin.latte.text)
      it
    } else {
      set text(catppuccin.latte.text)
      it
    }
  }

  set align(center)
  set par(justify: false)
  set table.vline(stroke: white + 2pt)
  set table.hline(stroke: white + 2pt)

  let f = if fill == auto {
    (x, y) => if y in header-rows {
      header-color.fill
    } else if calc.even(y - header-rows.len()) { even-color } else { odd-color }
  } else {
    fill
  }

  let c = (header, ..content.pos()).slice(if header == auto {1} else {0}, content.pos().len() + 1)

  cut-block(
    table(
      columns: columns,
      align: alignment,
      stroke: stroke,
      fill: f,
      gutter: -1pt,
      //..c
      // header,
      // ..content
    )
  )
}
