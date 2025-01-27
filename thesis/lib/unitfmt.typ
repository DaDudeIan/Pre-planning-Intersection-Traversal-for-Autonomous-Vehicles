
#let KiB(bytes) = bytes / 1024
#let MiB(bytes, decimals: 2) = {
  let value = bytes / 1024 / 1024
  $#value "MiB"$
}
#let GiB(bytes) = bytes / 1024 / 1024 / 1024


#let KB(bytes) = bytes / 1000
#let MB(bytes) = bytes / 1000 / 1000
#let GB(bytes) = bytes / 1000 / 1000 / 1000
