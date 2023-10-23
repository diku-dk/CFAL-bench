{-# LANGUAGE PatternSynonyms #-}
module Input where
import Data.Array.Accelerate
import Physics
import Prelude ()

gen_input :: Acc (Scalar Int) -> Acc (Vector Body)
gen_input n' = let n = the n' in generate 
  (I1 n) 
  (\(I1 i) -> let f = fromIntegral i in 
    Body 
      (Three (sin f) (cos f) (tan f)) -- position
      (sin (f+1.1)) -- mass
      (Three 0 0 0)) -- velocity

