#let stringify(text) = {
  let string = if type(text) == "string" {
    text
  } else if type(text) == "content" {
    repr(text).slice(1, -1) // remove square brackets
  } else {
    panic("Invalid type for text. Valid types are 'string' | 'content'")
  }

  return string
}

#let tokenize(string) = {
  stringify(string).trim().split(" ")
}

#let titlecase(string) = {
  let tokens = tokenize(stringify(string))

  tokens.map(
    word => {
      let chars = word.split("")
      (upper(chars.at(1)), ..chars.slice(2, -1)).join("") // join into a word again
    },
  ).join(" ") // join into a sentence again
}

#let camelcase(string) = {
  let tokens = tokenize(stringify(string))
  let first = tokens.first()
  let rest = tokens.slice(2, -1)
  .map(
    word => {
      let chars = word.split("")
      (upper(chars.at(1)), ..chars.slice(2, -1)).join("") // join into a word again
    },
  ).join("")

  first + rest
}

#let pascalcase(string) = titlecase(string).replace(" ", "")

#let joinwith(string, separator: " ") = {
  let tokens = tokenize(stringify(string))
  tokens.map(lower).join(separator)
}

#let snakecase = joinwith.with(separator: "_")
#let kebabcase = joinwith.with(separator: "-")
#let dotcase = joinwith.with(separator: ".")
#let slashcase = joinwith.with(separator: "/")

#let screaming-snakecase(string) = upper(snakecase(string))

#let screaming-kebabcase(string) = upper(kebabcase(string))
