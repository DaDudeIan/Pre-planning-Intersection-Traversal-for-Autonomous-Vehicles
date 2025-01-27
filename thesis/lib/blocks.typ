#import "catppuccin.typ": *

#let std-block = block.with(
  fill: catppuccin.latte.base,
  radius: 1em,
  inset: 0.75em,
  stroke: none,
  width: 100%,
  breakable: true,
)

#let cut-block = block.with(
  fill: none,
  radius: 1em,
  stroke: none,
  breakable: true,
  clip: true,
)

#let blocked(title: none, content, color: catppuccin.latte.base, height: auto, divider-stroke: white + 2pt) = {
  set align(left)
  std-block(fill: color, height: height)[
    #v(0.25em)
    #text(catppuccin.latte.text, size: 1.2em, weight: 900, title)
    // #v(-0.15em)

    #move(dx: -0.75em, dy: 0pt, line(length: 100% + 2 * 0.75em, stroke: divider-stroke))

    #content

    #v(0.5em)
  ]
}
