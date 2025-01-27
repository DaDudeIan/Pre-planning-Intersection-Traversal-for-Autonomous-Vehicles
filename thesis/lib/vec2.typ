#let add(a, b) = (a.at(0) + b.at(0), a.at(1) + b.at(1))
#let sub(a, b) = (a.at(0) - b.at(0), a.at(1) - b.at(1))
#let scale(k, a) = (a.at(0) * k, a.at(1) * k)

#let dot(a, b) = a.at(0) * b.at(0) + a.at(1) * b.at(1)

#let distance(a, b) = calc.sqrt(calc.pow(a.at(0) - b.at(0), 2) + calc.pow(a.at(1) - b.at(1), 2))
#let distance-squared(a, b) = calc.pow(a.at(0) - b.at(0), 2) + calc.pow(a.at(1) - b.at(1), 2)
#let norm(a) = calc.sqrt(calc.pow(a.at(0), 2) + calc.pow(a.at(1), 2))

#let normalize(a) = {
  let norm = norm(a)
  (a.at(0) / norm, a.at(1) / norm)
}

#let direction(a, b) = normalize(sub(b, a))

#let lerp(from, to, amount) = {
  let dir = sub(to, from)
  let incr = (dir.at(0) * amount, dir.at(1) * amount)
  add(from, incr)
}

#let from-polar(radius, angle) = (radius * calc.cos(angle), radius * calc.sin(angle))

#let normals(a) = {
  let (x, y) = a

  // (
  //   (y, -x),
  //   (-y, x)
  // )

  (
    clockwise: (y, -x),
    counter: (-y, x)
  )
}

#let projection-onto-line(point, a, b) = {
  // let x1 = point.at(0)
  // let y1 = point.at(1)
  let (x1, y1) = point
  let xp = (x1 + a * (y1 - b)) / (a * a + 1)
  (
    xp,
    a * xp + b
  )
}

#let line-from-line-segment(start, end) = {
  // let x1 = start.at(0)
  // let y1 = start.at(1)
  let (x1, y1) = start

  // let x2 = end.at(0)
  // let y2 = end.at(1)
  let (x2, y2) = end

  let a = (y2 - y1) / (x2 - x1)
  let b = y1 - (a * x1)

  (
    a: a,
    b: b
  )
}

#let rotate-z(p, theta: calc.pi) = {
  let (x, y) = p
  let cos = calc.cos
  let sin = calc.sin
  (
    cos(theta) * x  + -sin(theta) * y,
    sin(theta) * x + cos(theta) * y
  )
}

#let points-relative-from(start, ..points) = {
  // check the type of the first element in each tuple in the list points
  if type(points.pos().first().first()) == angle {
    points.pos().fold((start, ), (acc, point) => {
      let (angle, distance) = point
      acc + (add(acc.last(), from-polar(distance, angle)),)
    })
  } else {
    points.pos().fold((start, ), (acc, point) => {
      acc + (add(acc.last(), point),)
    })
  }
}

// #let points-relative-from-angle(start, ..points) = {
//   // but where a point is (angle, distance)
// }
