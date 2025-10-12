{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FlexibleInstances #-}
module Physics where
import Data.Array.Accelerate hiding (Vec)

data Vec = Vec_ Double Double Double
  deriving (Generic, Elt)

type Mass         = Double
type Position     = Vec
type Acceleration = Vec
type Velocity     = Vec
data PointMass    = PointMass_ Position Mass
  deriving (Generic, Elt)
data Body         = Body_ Position Mass Velocity
  deriving (Generic, Elt)

mkPatterns [''Vec, ''PointMass, ''Body]

instance Prelude.Num (Exp Vec) where
  Vec a b c + Vec x y z = Vec (a+x) (b+y) (c+z)
  Vec a b c - Vec x y z = Vec (a-x) (b-y) (c-z)
  Vec a b c * Vec x y z = Vec (a*x) (b*y) (c*z)
  negate (Vec a b c) = Vec (negate a) (negate b) (negate c)
  abs    (Vec a b c) = Vec (abs a)    (abs b)    (abs c)
  signum (Vec a b c) = Vec (signum a) (signum b) (signum c)
  fromInteger i = Vec (fromInteger i) (fromInteger i) (fromInteger i)

dot :: Exp Vec -> Exp Vec -> Exp Double
dot (Vec a b c) (Vec x y z) = a*x + b*y + c*z

scale :: Exp Double -> Exp Vec -> Exp Vec
scale s (Vec a b c) = Vec (s*a) (s*b) (s*c)

epsilon :: Exp Double
epsilon = 1e-9

pointmass :: Exp Body -> Exp PointMass
pointmass (Body p m _) = PointMass p m

accel :: Exp PointMass -> Exp PointMass -> Exp Velocity
accel (PointMass xpos _) (PointMass ypos ymass) =
  let r = ypos - xpos
      rsqr = dot r r + epsilon
      inv_dist = 1 / sqrt rsqr
      inv_dist3 = inv_dist * inv_dist * inv_dist
      s = ymass * inv_dist3
  in scale s r

advance_body :: Exp Double -> Exp Body -> Exp Acceleration -> Exp Body
advance_body time_step (Body pos mass vel) acc =
  let position = pos + scale time_step vel
      velocity = vel + scale time_step acc
  in Body position mass velocity
