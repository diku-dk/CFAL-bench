{-# LANGUAGE RankNTypes #-}
module Main where
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
-- import qualified Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion
import Criterion.Main

import Naive
import Flash_alg1
import Criterion.Types (Benchmark(Benchmark))

main :: IO ()
main = do
  -- check: these two values should be _very close_
  print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [8])
  print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [64])
  print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [1024])
  print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [16384])
  -- higher values for `m` crash Accelerate for some reason, even though the formula claims that these three should be fine:
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [32768])
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [65536])
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [131072])
  defaultMain [backend "CPU" CPU.runN
                -- , backend "GPU" GPU.runN]
                   ]
  where
    backend name runN
      = bgroup name
      $ map (testcase runN)
      $ (,,) <$> [512, 1024, 2048, 4096, 8192, 16384]
             <*> [64, 128]
             <*> [8, 64, 512, 2048, 8192, 16384]
    testcase runN (n,d,m) = bench ("n" ++ show n ++ ", d" ++ show d ++ ", m" ++ show m)
      $ nf (runN totalProgram) ( A.fromList A.Z [n]
                               , A.fromList A.Z [d]
                               , A.fromList A.Z [m])
