module Main where
import NBody
import Data.Array.Accelerate (fromList, Z(..))
import Data.Array.Accelerate.LLVM.Native as CPU
import Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion
import Criterion.Main

-- read input, run accelerate, benchmark
main :: IO ()
main = defaultMain [backend "CPU" CPU.runN, backend "GPU" GPU.runN]
  where
    backend s r = bgroup s $ map (size r) [1000,10000,100000]
    size r n = env (return (r nbody, fromList Z [0.1], fromList Z [n], fromList Z [10])) $ \ ~(p,dt,n,k) -> bench (show n) $ nf (p dt n) k
