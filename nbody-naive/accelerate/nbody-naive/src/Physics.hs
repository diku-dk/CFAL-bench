{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FlexibleInstances #-}
module Physics where
import Data.Array.Accelerate

data ThreeDoubles = Three_ {x_ ::Double, y_ :: Double, z_ :: Double}
  deriving (Generic, Elt)

type Mass = Double
type Position = ThreeDoubles
type Acceleration = ThreeDoubles
type Velocity = ThreeDoubles
data PointMass = PointMass_ {pmposition_ :: Position, pmmass_ :: Mass}
  deriving (Generic, Elt)
data Body = Body_ {bposition_ :: Position, bmass_ :: Mass, bvelocity_ :: Velocity}
  deriving (Generic, Elt)

mkPatterns [''ThreeDoubles, ''PointMass, ''Body]

instance Prelude.Num (Exp ThreeDoubles) where
  (+) = match \(Three a b c) (Three x y z) -> Three (a+x) (b+y) (c+z)
  (-) = match \(Three a b c) (Three x y z) -> Three (a-x) (b-y) (c-z)
  (*) = match \(Three a b c) (Three x y z) -> Three (a*x) (b*y) (c*z)
  negate = match \(Three a b c) -> Three (negate a) (negate b) (negate c)
  abs = match \(Three a b c) -> Three (abs a) (abs b) (abs c)
  signum = match \(Three a b c) -> Three (signum a) (signum b) (signum c)
  fromInteger i = Three (fromInteger i) (fromInteger i) (fromInteger i)

dot :: Exp ThreeDoubles -> Exp ThreeDoubles -> Exp Double
dot = match $ \(Three a b c) (Three x y z) -> a*x+b*y+c*z
scale :: Exp Double -> Exp ThreeDoubles -> Exp ThreeDoubles
scale s = match $ \(Three a b c) -> Three (s*a) (s*b) (s*c)

epsilon :: Exp Double
epsilon = constant 1e-9

pointmass :: Exp Body -> Exp PointMass
pointmass = match \case
  Body p m _ -> PointMass p m

accel :: Exp PointMass -> Exp PointMass -> Exp Velocity
accel x y =
  let r = pmposition y - pmposition x
      rsqr = dot r r + epsilon -- ???
      inv_dist = constant 1 / sqrt rsqr
      inv_dist3 = inv_dist * inv_dist * inv_dist
      s = pmmass y * inv_dist3
  in scale s r

advance_body :: Exp Double -> Exp Body -> Exp Acceleration -> Exp Body
advance_body time_step body acc =
  let position = bposition body + scale time_step (bvelocity body)
      velocity = bvelocity body + scale time_step acc
  in Body position (bmass body) velocity

