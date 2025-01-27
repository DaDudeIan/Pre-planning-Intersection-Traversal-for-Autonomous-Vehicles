#import "@preview/drafting:0.2.0": margin-note
#import "catppuccin.typ": *

#let note-gen(note, c: color.red, scope: "") = {
  let note = [
    #set text(size: 8pt)
    #set par(justify: false)
    #text(c, scope) \
    // #line(length: 100%, stroke: c + 0.5pt)
    #note
  ]
  margin-note(side: left, stroke: c + 1pt, note)
}

#let kristoffer = note-gen.with(c: catppuccin.latte.blue, scope: "Kristoffer")
#let k = kristoffer
#let kevork = note-gen.with(c: catppuccin.latte.yellow, scope: "Kevork")
#let ke = kevork
#let jens = note-gen.with(c: catppuccin.latte.teal, scope: "Jens")
#let j = jens
#let jonas = note-gen.with(c: catppuccin.latte.mauve, scope: "Jonas")
#let jo = jonas
#let layout = note-gen.with(c: catppuccin.latte.maroon, scope: "Layout")
#let l = layout
#let wording = note-gen.with(c: catppuccin.latte.green, scope: "Wordings")
#let w = wording
#let attention = note-gen.with(c: catppuccin.latte.peach, scope: "Attention")
#let a = attention

// #let krisoffer(note) = {
//   let c = catppuccin.latte.yellow
//   let note = [
//     #text(c, "Kristoffer")
//     #note
//   ]
//
//   margin-note(side: left, stroke: c + 1pt, note)
// }

// #let jens(note) = {
//   let c = catppuccin.latte.mauve
//   let note = [
//     #text(c, "Jens")
//     #note
//   ]
//
//   margin-note(side: right, stroke: c + 1pt, note)
// }
//
// #let jonas(note) = {
//   let c = catppuccin.latte.blue
//   let note = [
//     #text(c, "Jonas")
//     #note
//   ]
//
//   margin-note(side: right, stroke: c + 1pt, note)
// }
//
// #let layout(note) = {
//   let c = catppuccin.latte.maroon
//   let note = [
//     #text(c, "Layout")
//     #note
//   ]
//
//   margin-note(side: right, stroke: c + 1pt, note)
// }
//
// #let wording(note) = {
//   let c = catppuccin.latte.teal
//   let note = [
//     #text(c, "Layout")
//     #note
//   ]
//
//   margin-note(side: right, stroke: c + 1pt, note)
// }
