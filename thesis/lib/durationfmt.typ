

#let durationfmt(d) = {
  assert(type(d) == duration)

  let weeks = calc.floor(d.weeks())
  let years = calc.div-euclid(weeks, 52)
  let days = d.days() - 7 * weeks
  weeks = calc.rem(weeks, 52)

  let parts = ()

  if years > 0 {
    if years == 1 {
      parts.push([1 year])
    } else {
      parts.push([#years years])
    }
  }

  if weeks > 0 {
    if weeks == 1 {
      parts.push([1 week])
    } else {
      parts.push([#weeks weeks])
    }
  }

  if days > 0 {
    if days == 1 {
      parts.push([1 day])
    } else {
      parts.push([#days days])
    }
  }

  // parts

  if parts.len() == 1 {
    parts.first()
  } else if parts.len() == 2 {
    parts.first()
    [ and ]
    parts.last()
  } else if parts.len() > 2 {
    parts.slice(0, parts.len() - 1).join(", ")
    [ and ]
    parts.last()
  }

  // [ ago]
}

// #let jan1 = datetime(day: 1, month: 1, year: 2023)
// #let april1 = datetime(day: 1, month: 4, year: 2024)
// #let may1 = datetime(day: 2, month: 5, year: 2024)
// #let today = datetime.today()

// #durationfmt(today - jan1)
// #durationfmt(today - april1)
// #durationfmt(today - may1)



// #{ calc.div-euclid(104, 52)}
