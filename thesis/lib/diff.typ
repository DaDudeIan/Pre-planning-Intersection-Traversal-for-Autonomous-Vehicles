// #let values(dict) = for (k, v) in dict { ((k, v), ) }

#let diffarray(a, b) = {
  assert(type(a) == array, message: "expected `a` to have type 'array', got " + type(a))
  assert(type(b) == array, message: "expected `b` to have type 'array', got " + type(b))

  assert(a.len() == b.len(), message: "a.len() != b.len() => " + str(a.len()) + " != " + str(b.len()))

  a.zip(b).map(pair => {
    let x = pair.at(0)
    let y = pair.at(1)

    assert(type(x) == type(y))

    if type(x) == array {
      diffarray(x, y)
    } else if type(x) == dictionary {
      diffdict(x, y)
    } else {
      x == y
    }
  })

}

// #diffarray((1, 2, (3, 4)), (3, 2, (3, 5)))

#let diffdict(a, b) = {
  assert(type(a) == dictionary, message: "expected `a` to have type 'dictionary', got " + type(a))
  assert(type(b) == dictionary, message: "expected `b` to have type 'dictionary', got " + type(b))

  let diff = (:)

  for (k, va) in a {
    assert(k in b, message: "diffdict only works if both dicts contains, the same keys :(")

    let vb = b.at(k)
    assert(type(va) == type(vb))

    let d = if type(va) == array {
      vb.zip(va).map(pair => pair.at(0) != pair.at(1))
    } else if type(vb) == dictionary {
      diffdict(va, vb)
    } else {
      va != vb
    }

    diff.insert(k, d)
  }

  diff
}


// #let leafflatten(dict) = {
//   assert(type(dict) == dictionary, message: "expected `dict` to have type 'dictionary', got " + type(dict))
//
//   dict
//   .pairs()
//   .fold((:), (acc, pair) => {
//     let k = pair.at(0)
//     let v = pair.at(1)
//
//     if type(v) == dictionary {
//       acc + leafflatten(v)
//     } else {
//       acc.insert(k, v)
//       acc
//     }
//   })
// }
