#import "../../lib/mod.typ": *
= Nomenclature <nomenclature>

#par(first-line-indent: 0pt)[Some terminology and type setting used in this thesis may not be familiar to the reader, and are explained here for clarity.]

#term-table(
  [`monospace`], [Inline monospace text is used for code function names, variables, or parameters.],
  [`a.b`], [In inline monospace text, a period `.` is used to denote a method or property of an object. Can also be used outside of monospace text.],
  [`listing:<int>`], [A reference to a specific listing, where `<int>` represents a line number.],
  [`listing:<int>-<int>`], [Reference to a range of lines within a listing.],
  [`file.ext`],[A reference to a specific file of given file type.],
  [`file.ext:<func>`], [A reference to a specific function within a file.],
  [Stack], [Also called a suite, a stack is a collection of related functions.]
)

// consider:
// Parameter: A variable used to define a system or function, often denoted in italics or monospace.
// Argument: A value provided to a function or command.
// Type: The classification of data (e.g., integer, string, array).
// Array: An ordered collection of elements, typically of the same type.
// State: A stored value or set of values that can be accessed or modified during document processing..
// Grid: A layout structure for arranging elements in columns and rows.