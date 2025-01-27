#let binary(x, min-length: none) = {
  assert(type(x) == int, message: "expected `x` to have type `int`, but got " + type(x))

  let bits = ()

  while x != 0 {
    let is-odd = calc.rem(x, 2) == 1

    if is-odd {
      bits.push(1)
    } else {
      bits.push(0)
    }

    // Divide x by 2 to process the next bit
    x = calc.floor(x / 2)
  }

  // If no bits were added (i.e., x was 0), add a single 0 bit
  if bits.len() == 0 {
    bits.push(0)
  }

  // Reverse the bits to get the correct binary representation
  let bits = bits.rev()


  if min-length != none and bits.len() < min-length {
    bits = range(min-length - bits.len()).map(_ => 0) + bits
  }

  bits
}

#let hamming-distance(a, b, n: 32) = {
  binary(a, min-length: n)
    .zip(binary(b, min-length: n))
    .fold(0, (acc, pair) =>
    acc + if pair.at(0) != pair.at(1) { 1 } else { 0 }
  )
}

#let popcount(x) = binary(x).sum()
