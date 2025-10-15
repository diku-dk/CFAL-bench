{-# LANGUAGE ViewPatterns #-}
module Input where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Data.Bits
import Prelude ()


mkInput :: Acc (Scalar (Int, Int)) -> Acc (Matrix Float, Matrix Float, Matrix Float)
mkInput (the -> T2 nN d) =
  let ones = generate (I2 nN d) (\_ -> 1.0)
  in T3 ones ones ones

-- Given an output computed from mkInput, return (target statistic, actual
-- statistic). Equality of the two values does not imply correctness of the
-- implementation, but inequality does show incorrectness.
checkCorrectness :: Acc (Matrix Float) -> Acc (Scalar Float, Scalar Float)
checkCorrectness o =
  let I2 nN d = shape o
      target = sqrt (toFloating nN * toFloating d)
      result = map sqrt $ sum $ sum $ map (\x -> x*x) o
  in T2 (unit target) result

mkRandomInput :: Acc (Scalar Int) -> Acc (Scalar (Int, Int)) -> Acc (Matrix Float, Matrix Float, Matrix Float)
mkRandomInput (the -> seed) (the -> T2 nN d) =
  let sh = I2 nN d
      gen salt = generate sh (randomize (seed + salt) . toIndex sh)
  in T3 (gen 123) (gen 456) (gen 789)

randomize :: Exp Int -> Exp Int -> Exp Float
randomize seed x = abs (sin (A.fromIntegral d))
  where
    a = x * 17 + seed
    b = a `xor` (a `shiftL` 19)
    c = b `xor` (b `shiftR` 5)
    d = c `xor` (c `shiftL` 7)
