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
      (Vec (sin f) (cos f) (tan f)) -- position
      (sin (f+1.1)) -- mass
      (Vec 0 0 0)) -- velocity

