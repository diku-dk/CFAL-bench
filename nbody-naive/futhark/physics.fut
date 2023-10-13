import "lib/github.com/athas/vector/vspace"

module vec3 = mk_vspace_3d f64

type mass = f64
type position = vec3.vector
type acceleration = vec3.vector
type velocity = vec3.vector
type pointmass = {position: position,
                  mass: mass}
type body = {position: position,
             mass: mass,
             velocity: velocity}

def EPSILON : f64 = 1e-9

def pointmass ({position, mass, velocity=_}: body) : pointmass =
  {position, mass}

def accel (x: pointmass) (y: pointmass): velocity =
  let r = vec3.(y.position - x.position)
  let rsqr = vec3.dot r r + EPSILON
  let inv_dist = 1 / f64.sqrt rsqr
  let inv_dist3 = inv_dist * inv_dist * inv_dist
  let s = y.mass * inv_dist3
  in vec3.scale s r

def advance_body (time_step: f64) (body: body) (acc: acceleration): body =
  let position = vec3.(body.position + scale time_step body.velocity)
  let velocity = vec3.(body.velocity + scale time_step acc)
  in {position, mass=body.mass, velocity}
