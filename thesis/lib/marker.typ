#import "catppuccin.typ": catppuccin
#let accent = catppuccin.latte.lavender
#let arrow = (
  single: {
    place(
      left,
      line(stroke: (paint: auto, thickness: 2pt, cap: "round"), start: (0em, 0.2em), end: (0.2em, 0.4em))
    )
    place(
      left,
      line(stroke: (paint: auto, thickness: 2pt, cap: "round"), start: (0em, 0.6em), end: (0.2em, 0.4em))
    )
    h(0.5em)
  },
  double: {
    place(
      dx: 0em,
      dy: 0em,
      line(stroke: (paint: auto, thickness: 2pt, cap: "round"), start: (0em, 0.2em), end: (0.2em, 0.4em))
    )
    place(
      dx: 0em,
      dy: 0em,
      line(stroke: (paint: auto, thickness: 2pt, cap: "round"), start: (0em, 0.6em), end: (0.2em, 0.4em))
    )
    place(
      dx: 0.4em,
      dy: 0em,
      line(stroke: (paint: auto, thickness: 2pt, cap: "round"), start: (0em, 0.2em), end: (0.2em, 0.4em))
    )
    place(
      dx: 0.4em,
      dy: 0em,
      line(stroke: (paint: auto, thickness: 2pt, cap: "round"), start: (0em, 0.6em), end: (0.2em, 0.4em))
    )
    h(0.75em)
  }
)
