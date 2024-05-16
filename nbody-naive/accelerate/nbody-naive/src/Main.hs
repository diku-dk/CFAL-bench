{-# LANGUAGE TypeApplications #-}
module Main where
import NBody
import Data.Array.Accelerate
import Data.Array.Accelerate.LLVM.Native as CPU
-- import Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion
import Criterion.Main
import Debug.Trace
import Physics (pointmass)
import Input (gen_input)

-- read input, run accelerate, benchmark
main :: IO ()
main = do
  -- putStrLn $ test @UniformScheduleFun @CPU.NativeKernel nbody
  -- print $ runN @CPU.Native nbody (fromList Z [0.1]) (fromList Z [1000]) (fromList Z [10])


  defaultMain [backend "CPU" $ runN @CPU.Native ] --, backend "GPU" GPU.runN]
  where
    backend s r = bgroup s $ Prelude.map (size r) [1000,2000,4000]
    size r n = env (return (r nbody, fromList Z [0.1], fromList Z [n], fromList Z [10])) $ \ ~(p,dt,n,k) -> bench (show n) $ nf (p dt n) k
