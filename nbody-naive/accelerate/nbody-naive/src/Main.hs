{-# LANGUAGE BangPatterns #-}
module Main where
import NBody
import Data.Array.Accelerate (fromList, Z(..))
import Data.Array.Accelerate.LLVM.Native as CPU
import Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion
import Criterion.Main
import Criterion.Measurement
import Criterion.Measurement.Types (Measured(measTime))

-- read input, run accelerate, benchmark
main :: IO ()
main = do
  let !cpu = CPU.runN nbody
  let !gpu = GPU.runN nbody
  mapM_ (test cpu) 
    [(1000, 100000)
    ,(10000,  1000)
    ,(100000,   10)]
  mapM_ (test gpu) 
    [(1000, 100000)
    ,(10000,  1000)
    ,(100000,   10)]
  
  -- defaultMainWith (defaultConfig { timeLimit = 30}) [backend "CPU" CPU.runN, backend "GPU" GPU.runN]
  where
    -- backend s r = bgroup s $ map (size r) [(1000, 1000)
    --                                       ,(1000, 10000)
    --                                       ,(1000, 100000)
    --                                       ,(10000,  1000)
    --                                       ,(100000,   10)]
    -- size r (n,t) = env (return (r nbody, fromList Z [0.1], fromList Z [n], fromList Z [t])) $ \ ~(p,dt,n,k) -> bench (show (n,t)) $ nf (p dt n) k
    test p (n,t) = do
      print (n,t)
      (measured, endtime) <- measure (nf (p (fromList Z [0.1]) (fromList Z [n])) (fromList Z [t])) 2
      print (secs $ measTime measured)
