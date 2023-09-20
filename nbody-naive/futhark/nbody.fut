import "physics"

def calc_accels [n] (epsilon: f64) (bodies: [n]pointmass): [n]acceleration =
  let move (body: pointmass) =
    let accels = map (accel epsilon body) bodies
    in reduce_comm (vec3.+) {x=0f64, y=0f64, z=0f64} accels
  in map move bodies

def advance [n] (epsilon: f64) (time_step: f64) (bodies: [n]body): [n]body =
  map2 (advance_body time_step) bodies
       (calc_accels epsilon (map pointmass bodies))

def epsilon : f64 = 1e-9

entry mk_positions (n: i64) =
  let pos x = [f64.sin x, f64.cos x, f64.tan x]
  let positions = tabulate n (f64.i64 >-> pos)
  in positions

entry mk_masses (n: i64) =
  tabulate n (f64.i64 >-> (+1.1) >-> f64.sin)

entry main [n]
        (k: i32)
        (dt: f64)
        (positions: [n][3]f64)
        (masses: [n]f64) =
  let mk position mass =
    {position={x=position[0],y=position[1],z=position[2]},
     mass,
     velocity={x=0,y=0,z=0}}
  let unmk p = [p.position.x, p.position.y, p.position.z]
  in map2 mk positions masses
     |> iterate k (advance epsilon dt)
     |> map unmk

-- ==
-- entry: main
-- "k=10, n=1000"   script input { (10i32, 0.1f64, mk_positions 1000,   mk_masses 1000) }
-- "k=10, n=10000"  script input { (10i32, 0.1f64, mk_positions 10000,  mk_masses 10000) }
-- "k=10, n=100000" script input { (10i32, 0.1f64, mk_positions 100000, mk_masses 100000) }
