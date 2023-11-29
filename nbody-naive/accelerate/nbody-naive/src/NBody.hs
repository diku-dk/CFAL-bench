module NBody where
import Data.Array.Accelerate
import Physics
import Input
import Prelude ()

-- The replicates get fused away
calc_accels :: Acc (Vector PointMass) -> Acc (Vector Acceleration)
calc_accels bodies =
  let Z_ ::. n = shape bodies
      bodies'  = replicate (Z_ ::. n ::. All_) bodies
      bodies'' = replicate (Z_ ::. All_ ::. n) bodies
  in fold (+) (fromInteger 0) $ zipWith accel bodies' bodies''

step :: Exp Double -> Acc (Vector Body) -> Acc (Vector Body)
step dt bodies = zipWith (advance_body dt) bodies (calc_accels $ map pointmass bodies)

nbody :: Acc (Scalar Double) -> Acc (Scalar Int) -> Acc (Scalar Int) -> Acc (Vector Body)
nbody dt n k = afst $ awhile (map (< the k) . asnd) (\(T2 x i) -> T2 (step (the dt) x) (map (+1) i)) (T2 (gen_input n) (unit $ constant 0))
