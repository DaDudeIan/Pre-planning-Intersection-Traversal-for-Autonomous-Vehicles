#let aos(dict) = {
  for (k, v) in dict {
    assert(type(v) == array)
  }

  let array-lens = dict.values().map(arr => arr.len())
  assert(array-lens.all(len => len == array-lens.at(0)))

  let output = ()
  for i in range(array-lens.at(0)) {
    let record = (:)
    for (k, v) in dict {
      record.insert(k, v.at(i))
    }
    output.push(record)
  }

  return output
}

#let soa(dictionaries) = {
  assert(type(dictionaries) == array)

  for dict in dictionaries {
    assert(type(dict) == dictionary)
  }

  let keys = dictionaries.at(0).keys()
  assert(dictionaries.map(dict => dict.keys()).all(k => k == keys))

  let output = keys.fold((:), (acc, k) => {
    acc.insert(k, ())
    acc
  })


  for dict in dictionaries {
    for (k, v) in dict {
      let arr = output.at(k)
      arr.push(v)
      output.insert(k, arr)
    }
  }

  return output
}


// #aos(soa(((a: 2, b: 3), (a: 4, b: 5))))
