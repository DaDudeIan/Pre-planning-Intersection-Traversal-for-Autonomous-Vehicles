#let leafmap(dict, f) = {
  assert(type(dict) == dictionary, message: "expected `dict` to have type 'dictionary', got " + type(dict))
  assert(type(f) == function, message: "expected `f` to have type `function` with signature (k: string, v: any) => any, got " + type(f))
  dict
    .pairs()
    .fold((:), (acc, pair) => {
      let k = pair.at(0)
      let v = pair.at(1)
      let v_mapped = if type(v) == dictionary {
        leafmap(v, f)
      } else if type(v) == array {
        v.map(it => f(k, it))
      } else {
        f(k, v)
      }
      acc.insert(k, v_mapped)
      acc
    })
}

#let leafassoc(dict, f) = {
  assert(type(dict) == dictionary, message: "expected `dict` to have type 'dictionary', got " + type(dict))
  assert(type(f) == function, message: "expected `f` to have type `function` with signature (k: string, v: any) => any, got " + type(f))
  dict
    .pairs()
    .fold((:), (acc, pair) => {
      let k = pair.at(0)
      let v = pair.at(1)
      let v_mapped = if type(v) == dictionary {
        leafassoc(v, f)
      } else if type(v) == array {
        v.map(it => (it, f(k, it)))
      } else {
        (v, f(k, v))
      }
      acc.insert(k, v_mapped)
      acc
    })
}

#let leafzip(a_dict, b_dict) = {
  assert(type(a_dict) == dictionary, message: "expected `a_dict` to have type 'dictionary', got " + type(a_dict))

  assert(type(b_dict) == dictionary, message: "expected `b_dict` to have type 'dictionary', got " + type(b_dict))


  a_dict
  .pairs()
  .fold((:), (acc, pair) => {
    let k = pair.at(0)
    let va = pair.at(1)

    assert(k in b_dict, message: "diffdict only works if both dicts contains, the same keys :(")
    let vb = b_dict.at(k)
    let v_zipped = if type(va) == array {
      assert(va.len() == vb.len())
      va.zip(vb)
    } else if type(va) == dictionary {
      leafzip(va, vb)
    } else {
      (va, vb)
    }

    acc.insert(k, v_zipped)
    acc
  })
}


#let leafflatten(dict) = {
  assert(type(dict) == dictionary, message: "expected `dict` to have type 'dictionary', got " + type(dict))

  dict
  .pairs()
  .fold((:), (acc, pair) => {
    let k = pair.at(0)
    let v = pair.at(1)

    if type(v) == dictionary {
      acc + leafflatten(v)
    } else {
      acc.insert(k, v)
      acc
    }
  })
}
