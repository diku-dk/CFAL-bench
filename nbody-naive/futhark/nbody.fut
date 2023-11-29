type vec = {x: f64, y: f64, z: f64}

def vecadd (a: vec) (b: vec) =
  {x = a.x+b.x, y = a.y+b.y, z = a.z+b.z}
def vecsub (a: vec) (b: vec) =
  {x = a.x-b.x, y = a.y-b.y, z = a.z-b.z}
def vecscale (s: f64) (a: vec) =
  {x = s * a.x, y = s * a.y, z = s * a.z}
def dot (a: vec) (b: vec) =
  a.x*b.x + a.y*b.y + a.z*b.z

type body = {pos: vec, vel: vec, mass: f64}

def EPSILON : f64 = 1e-9

def accel (x: body) (y: body): vec =
  let r = y.pos `vecsub` x.pos
  let rsqr = dot r r + EPSILON
  let inv_dist = 1 / f64.sqrt rsqr
  let inv_dist3 = inv_dist * inv_dist * inv_dist
  let s = y.mass * inv_dist3
  in vecscale s r

def advance_body (dt: f64) (body: body) (acc: vec): body =
  body with pos = vecadd body.pos (vecscale dt body.vel)
       with vel = vecadd body.vel (vecscale dt acc)

def calc_accels [n] (bodies: [n]body): [n]vec =
  let move (body: body) =
    let accels = map (accel body) bodies
    in reduce_comm vecadd {x=0, y=0, z=0} accels
  in map move bodies

def step [n] (dt: f64) (bodies: [n]body): [n]body =
  map2 (advance_body dt) bodies
       (calc_accels bodies)

def nbody [n] (k: i32) (dt: f64) (bodies: [n]body): [n]body =
  loop bodies' = bodies for _i < k do step dt bodies'

-- Everything below is boilerplate for benchmarking.

entry main [n] (k: i32) (dt: f64) (positions: [n][3]f64) (masses: [n]f64) =
  let mk position mass =
    {pos={x=position[0], y=position[1], z=position[2]},
     mass,
     vel={x=0, y=0, z=0}}
  let unmk p = [p.pos.x, p.pos.y, p.pos.z]
  in map2 mk positions masses
     |> nbody k dt
     |> map unmk

entry mk_positions (n: i64) =
  let pos x = [f64.sin x, f64.cos x, f64.tan x]
  let positions = tabulate n (f64.i64 >-> pos)
  in positions

entry mk_masses (n: i64) =
  tabulate n (f64.i64 >-> (+1.1) >-> f64.sin)

-- ==
-- entry: main
-- "k=10, n=1000"   script input { (10i32, 0.1f64, mk_positions 1000,   mk_masses 1000) }
-- "k=10, n=10000"  script input { (10i32, 0.1f64, mk_positions 10000,  mk_masses 10000) }
-- "k=10, n=100000" script input { (10i32, 0.1f64, mk_positions 100000, mk_masses 100000) }
