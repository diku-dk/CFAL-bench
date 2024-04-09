{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE PatternSynonyms #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FlexibleInstances #-}
module Physics where
import Data.Array.Accelerate

data ThreeDoubles = Three_ Double Double Double
  deriving (Generic, Elt, Show)

type Mass = Double
type Position = ThreeDoubles
type Acceleration = ThreeDoubles
type Velocity = ThreeDoubles
data PointMass = PointMass_ Position Mass
  deriving (Generic, Elt, Show)
data Body = Body_ Position Mass Velocity
  deriving (Generic, Elt, Show)

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
accel = match \(PointMass xpos _) (PointMass ypos ymass) ->
  let r = ypos - xpos
      rsqr = dot r r + epsilon -- ???
      inv_dist = constant 1 / sqrt rsqr
      inv_dist3 = inv_dist * inv_dist * inv_dist
      s = ymass * inv_dist3
  in scale s r

advance_body :: Exp Double -> Exp Body -> Exp Acceleration -> Exp Body
advance_body = match $ \time_step (Body pos mass vel) acc ->
  let position = pos + scale time_step vel
      velocity = vel + scale time_step acc
  in Body position mass velocity

