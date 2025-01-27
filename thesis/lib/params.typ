#import "catppuccin.typ": *
#import "lib.typ": *
#import "diff.typ": diffdict
#import "dict.typ": leafmap, leafzip, leafflatten
#import "equation.typ"
#import "seq.typ": fib


#let seeds = (0, 31, 227, 252, 805)
#let na = $N "/" A$


#let syms = (
  delta_t: $Delta_t$,
  m_r: $M_E$,
  m_i: $M_I$,
  sigma_d: $sigma_d$,
  sigma_p: $sigma_p$,
  sigma_r: $sigma_i$,
  sigma_o: $sigma_o$,
  sigma_t: $sigma_t$,
  radius: $C_("radius")$,
  r_r: $r_R$,
  speed: $abs(v_0)$,
  n_r: $N_R$,
  s: $s e e d$,
  comms-radius: $r_("comms")$,
  comms-failure-prob: $gamma$,
  variable-temporal-dist: $t_(K-1)$,
  lookahead-multiple: $l_m$,
  interrobot-safety-distance: $d_r$,
  variables: $|V|$,
  max-iterations: $N_"RRT"$,
  step-size: $s$,
  collision-radius: $r_C$,
  neighbourhood-radius: $r_N$,
)

// let params = (
#let circle = (
  // sim: (
  // ),
    factor: (
      sigma_d: $1m$,
      sigma_p: $1 times 10^(-15)m$,
      sigma_r: $0.005m$,
      sigma_o: $0.005m$,
      sigma_t: na,
      interrobot-safety-distance: $2.2 times C_("radius")$,
    ),
  gbp: (
    delta_t: $0.1$,
    m_r: $10$,
    m_i: $50$,
    // S_r: $2.2$,
    comms-failure-prob: $0%$,
    // variable-temporal-dist: {let v = 2 * 50 / 15; $#v$},
    // variable-temporal-dist: $6.67s^*$, // 2 * 50m / 15m/s
    variable-temporal-dist: ${5, 13.33}s^*$, // 2 * 50m / 15m/s
  // variables: todo[...],
  lookahead-multiple: ${1, 3}^*$,
  ),
  env: (
    radius: $50m$,
    r_r: $tilde.op cal(U)(2,3) m$,
    comms-radius: $50m$,
    speed: $15m"/"s$,
    n_r: ${5, 10, ..., 50}$,

    // s: $2^*$,
  s: equation.as-set(seeds),
  ),
)

#let clear-circle = (
  // sim: (
  // ),
    factor: (
      sigma_d: $1m$,
      sigma_p: $1 times 10^(-15)m$,
      sigma_r: $0.005m$,
      sigma_o: $0.005m$,
      sigma_t: na,
      interrobot-safety-distance: $2.2 times C_("radius")$,
    ),
  gbp: (
    delta_t: $0.1$,
    m_r: $10$,
    m_i: $50$,
    // S_r: $2.2$,
    comms-failure-prob: $0%$,
    // variable-temporal-dist: todo[...],
    // variable-temporal-dist: $6.67s^*$, // 2 * 50m / 15m/s
    variable-temporal-dist: $5s^*$, // 2 * 50m / 15m/s
    // variables: todo[...],
    lookahead-multiple: $3^*$,
  ),
  env: (
    radius: $50m$,
    r_r: $tilde.op cal(U)(2,3) m$,
    comms-radius: $50m$,
    speed: $15m"/"s$,
    n_r: ${5, 10, ..., 50}$,
    // s: $2^*$,
  s: equation.as-set(seeds),
  ),
)

#let varying-network-connectivity = (
    factor: (
      sigma_d: $1m$,
      sigma_p: $1 times 10^(-15)m$,
      sigma_r: $0.005m$,
      sigma_o: $0.005m$,
      sigma_t: na,
      interrobot-safety-distance: $2.2 times C_("radius")$,
    ),
  gbp: (
    delta_t: $0.1$,
    m_r: $10$,
    m_i: $50$,
    // S_r: $2.2$,
    comms-failure-prob: $0%$,
    // variable-temporal-dist: todo[...],
    // variable-temporal-dist: $6.67s^*$, // 2 * 50m / 15m/s
    variable-temporal-dist: ${5, 13.33}s^*$, // 2 * 50m / 15m/s
    // variables: todo[...],
    lookahead-multiple: $3^*$,
  ),
  env: (
    radius: $50m$,
    r_r: $tilde.op cal(U)(2,3) m$,
    comms-radius: ${20, 40, ..., 80}m$,
    speed: $15m"/"s$,
    n_r: $30$,
    // s: ${0, 32, 64, 128, 255}^*$,
  s: equation.as-set(seeds),
  ),
)



#let junction = (
  // sim: (
  // ),
    factor: (
      sigma_d: $0.5m$,
      sigma_p: $1 times 10^(-15)m$,
      sigma_r: $0.005m$,
      sigma_o: $0.005m$,
      sigma_t: na,
      interrobot-safety-distance: $2.2 times C_("radius")$,
    ),
  gbp: (
    delta_t: $0.1$,
    m_r: $10$,
    m_i: $50$,
    // S_r: $2.2$,
    comms-failure-prob: $0%$,
    variable-temporal-dist: $2s$,
    // variables: todo[...],
    lookahead-multiple: $3^*$,
  ),
  env: (
    radius: $N "/" A$,
    r_r: $2m$,
    comms-radius: $50m$,
    speed: $15m"/"s$,
    // n_r: na,
    n_r: $Q_("in") times 50 s$,
    // n_r: ${5, 10, ..., 50}$,
    // s: $2^*$,
  s: equation.as-set(seeds),
  ),
)

#let communications-failure = (
  // sim: (
  // ),
    factor: (
      sigma_d: $1$,
      sigma_p: $1 times 10^(-15)$,
      sigma_r: $0.005$,
      sigma_o: $0.005$,
      sigma_t: na,
      interrobot-safety-distance: $2.2 times C_("radius")$,
    ),
  gbp: (
    delta_t: $0.1$,
    m_r: $10$,
    m_i: $50$,
    comms-failure-prob: ${0, 10, ..., 90}%$,
    // variable-temporal-dist: todo[...],
    // variable-temporal-dist: $6.67s^*$, // 2 * 50m / 15m/s
    variable-temporal-dist: ${5, 13.33}s^*$, // 2 * 50m / 15m/s
    lookahead-multiple: $3^*$,
    // variables: todo[...],
    // S_r: $2.2$,
  ),
  env: (
    radius: $50m$,
    r_r: $tilde.op cal(U)(2,3) m$,
    comms-radius: $50m$,
    n_r: $21$,
    speed: ${10, 15}m"/"s$,
    // s: ${0, 32, 64, 128, 255}^*$,
  s: equation.as-set(seeds),
  ),
)


// sigma-pose-fixed        = 0.0000000000000010000000036274937
// sigma-factor-dynamics   = 0.10000000149011612
// sigma-factor-interrobot = 0.009999999776482582
// sigma-factor-obstacle   = 0.009999999776482582
// sigma-factor-tracking   = 0.15000000596046448
// lookahead-multiple      = 3

#let solo-gp = (
  // sim: (
  // ),
    factor: (
      sigma_d: $0.1$,
      sigma_p: $1 times 10^(-15)$,
      sigma_r: $0.005$,
      sigma_o: $0.005$,
      sigma_t: $0.15$,
      interrobot-safety-distance: $2.2 times C_("radius")$,
    ),
  gbp: (
    delta_t: $0.1$,
    m_r: $10$,
    m_i: $10$,
    comms-failure-prob: $0%$,
    // variable-temporal-dist: todo[...],
    // variable-temporal-dist: $6.67s^*$, // 2 * 50m / 15m/s
    variable-temporal-dist: $5s$, // 2 * 50m / 15m/s
    lookahead-multiple: $3^*$,
    // variables: todo[...],
    // S_r: $2.2$,
  ),
  env: (
    radius: na,
    r_r: $2 m$,
    comms-radius: na,
    n_r: $1$,
    speed: $7m"/"s$,
    // s: ${0, 32, 64, 128, 255}^*$,
  s: equation.as-set(seeds),
  ),
)


// [gbp]
// sigma-pose-fixed = 0.0000000000000010000000036274937
// sigma-factor-dynamics = 0.10000000149011612
// sigma-factor-interrobot = 0.009999999776482582
// sigma-factor-obstacle = 0.009999999776482582
// sigma-factor-tracking = 0.5
// lookahead-multiple = 3
// variables = 10

#let collaborative-gp = (
  // sim: (
  // ),
    factor: (
      sigma_d: $0.1$,
      sigma_p: $1 times 10^(-15)$,
      sigma_r: $0.005$,
      sigma_o: $0.005$,
      sigma_t: ${0.15, 0.5}$,
      interrobot-safety-distance: $4 times C_("radius")$,
    ),
  gbp: (
    delta_t: $0.1$,
    m_r: $10$,
    m_i: $10$,
    comms-failure-prob: $0$,
    // variable-temporal-dist: todo[...],
    // variable-temporal-dist: $6.67s^*$, // 2 * 50m / 15m/s
    variable-temporal-dist: $5s$, // 2 * 50m / 15m/s
    lookahead-multiple: $3^*$,
    // variables: todo[...],
    // S_r: $2.2$,
  ),
  env: (
    radius: na,
    r_r: $2m$,
    comms-radius: $20m$,
    n_r: $100$, // 10 * 10
    speed: $7m"/"s$,
    // s: ${0, 32, 64, 128, 255}^*$,
  s: equation.as-set(seeds),
  ),
)

#let internals = fib(11).slice(2)
#let externals = internals

#let iteration-amount = (
  // sim: (
  // ),
    factor: (
      sigma_d: $1$,
      sigma_p: $1 times 10^(-15)$,
      sigma_r: $0.005$,
      sigma_o: $0.005$,
      sigma_t: na,
      interrobot-safety-distance: $2.5 times C_("radius")$,
    ),
  gbp: (
    delta_t: $0.1$,
    m_r: equation.as-set(externals),
    m_i: equation.as-set(internals),
    comms-failure-prob: $0%$,
    // variable-temporal-dist: todo[...],
    // variable-temporal-dist: $6.67s^*$, // 2 * 50m / 15m/s
    variable-temporal-dist: $5s^*$, // 2 * 50m / 15m/s
    lookahead-multiple: $3^*$,
    // variables: todo[...],
    // S_r: $2.2$,
  ),
  env: (
    radius: $50m$,
    r_r: $tilde.op cal(U)(2,3) m$,
    comms-radius: $50m$,
    n_r: $25$,
    speed: $10m"/"s$,
    // s: ${0, 32, 64, 128, 255}^*$,
  s: equation.as-set(seeds),
  ),
)

#let iteration-schedules = (
  // sim: (
  // ),
    factor: (
      sigma_d: $1$,
      sigma_p: $1 times 10^(-15)$,
      sigma_r: $0.005$,
      sigma_o: $0.005$,
      sigma_t: na,
      interrobot-safety-distance: $2.5 times C_("radius")$,
    ),
  gbp: (
    delta_t: $0.1$,
    m_r: ${5, 10, 15, 20, 25}$,
    m_i: $50$,
    comms-failure-prob: $0%$,
    // variable-temporal-dist: todo[...],
    // variable-temporal-dist: $6.67s^*$, // 2 * 50m / 15m/s
    variable-temporal-dist: $7s^*$, // 2 * 50m / 15m/s
    lookahead-multiple: $3^*$,
    // variables: todo[...],
    // S_r: $2.2$,
  ),
  env: (
    radius: $50m$,
    r_r: $tilde.op cal(U)(2,3) m$,
    comms-radius: $50m$,
    n_r: $30$,
    speed: $10m"/"s$,
    // s: ${0, 32, 64, 128, 255}^*$,
  s: equation.as-set(seeds),
  ),
)
    // params.tabular(params.communications-failure.env, previous: params.circle.env, title: [Environment]),
    // params.tabular(params.communications-failure.gbp, previous: params.circle.gbp, title: [GBP Algorithm], extra-rows: 0),
    // params.tabular(params.communications-failure.factor, previous: params.circle.factor, title: [Factor Settings]),



#let make-rows(subdict, previous: none) = {
  let diff = if previous == none {
    leafmap(subdict, (k, v) => false) } else {
    diffdict(previous, subdict)
  }

  assert(type(diff) == dictionary, message: "expected `diff` to have type 'dictionary', got " + type(diff))

  let dict = leafzip(subdict, diff)
  let dict_flattened = leafflatten(dict)
  for (k, pair) in dict_flattened {
    let v = pair.at(0)
    let different = pair.at(1)

    let k = syms.at(k)
    if different {
      (k, text(theme.peach, v))
    } else {
      (k, v)
    }
  }

  // let pair-list = subdict.pairs().filter(it => {
  //   not type(it.at(1)) == dictionary
  // }).map(it => {
  //   let key = it.at(0)
  //   let val = it.at(1)
  //   (syms.at(key), val)
  // })
  //
  // pair-list.flatten()
}

#let tabular(subdict, title: none, extra-rows: 0, previous: none) = {
  let header = table.header(
    [Param], [Value]
  )
  let header-rows = (0,)

  if header != none {
    header-rows = (0, 1)
    header = table.header(
      table.cell(colspan: 2, align: center, title),
      [Param], [Value]
    )
  }

  show table.cell : it => {
    if it.y in header-rows {
      set text(theme.text)
      strong(it)
    } else if calc.even(it.y) {
      set text(theme.text)
      strong(it)
    } else {
      set text(theme.text)
      it
    }
  }

  set align(center)

  cut-block(
    table(
      columns: (1fr, auto),
      align: (x, y) => (left, right).at(x) + horizon,
      stroke: none,
      header,
      fill: (x, y) => if y in (0, 1) {
        theme.lavender.lighten(50%)
      } else if calc.even(y) { theme.crust } else { theme.mantle },
      gutter: -1pt,
      ..make-rows(subdict, previous: previous),
      ..rep((" ", " "), extra-rows).flatten()
    )
  )
}
