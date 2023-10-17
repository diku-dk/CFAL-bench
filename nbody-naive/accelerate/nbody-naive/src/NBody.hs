module NBody where
import Data.Array.Accelerate
import Physics
import Prelude ()

-- is this the cleanest way? Futhark does `map (\b -> fold (+) 0 (map (accel b) bodies))`
calc_accels :: Acc (Vector PointMass) -> Acc (Vector Acceleration)
calc_accels bodies =
  let Z_ ::. n = shape bodies
      bodies' = replicate (Z_ ::. n ::. All_) bodies
      bodies'' = transpose bodies'
  in fold (+) (fromInteger 0) $ zipWith accel bodies' bodies''

step :: Exp Double -> Acc (Vector Body) -> Acc (Vector Body)
step dt bodies = zipWith (advance_body dt) bodies (calc_accels $ map pointmass bodies)
