#let fib(n) = {
  let a = 0
  let b = 1
  for _ in range(1, n) {
    (a,)
    let temp = b
    b = a + b
    a = temp
  }
}
