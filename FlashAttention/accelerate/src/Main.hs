{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE BangPatterns #-}
module Main where
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion
import Criterion.Main

import Naive
import Flash_alg1
import Criterion.Types (Benchmark(Benchmark))
import Criterion.Measurement
import Criterion.Measurement.Types (Measured(measTime))

main :: IO ()
main = do
  -- -- check: these two values should be _very close_
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [8])
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [64])
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [1024])
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [16384])
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [32768])
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [65536])
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [131072])
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [2^18])
  -- print $ CPU.runN check (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [2^19])
  --putStrLn "20"
  --print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^20])
  --print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^21])
  --print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^22])
  --print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^23])
  --print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^24])
  --putStrLn "25"
  --print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^25])
  --print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^26])
  --print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^27])
  --print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^28])
  --print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^29])
  let !cpu = CPU.runN totalProgram
  print $ A.arraySize $ cpu (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [16384])
  backend "CPU" cpu

  where
    backend name p
      = do
        print name
        mapM_ (testcase p)
          -- $ map (\(a,b)->(b,a)) 
          $ [(16384,64,2^22),(32768,64,2^23),(8192,128,2^22),(16384,128,2^23)]
            --concatMap (\(d,n) -> [(n,d,m) | m <- [2^22,2^23]]) 
            --[(64,16384),(64,32768),(128,8192),(128,16384)]
      -- $ (,,) <$> [512] --, 1024, 2048, 4096, 8192, 16384]
      --        <*> [64] --, 128]
      --        <*> [8] --, 64, 512, 2048, 8192, 16384]
    testcase p (n,d,m) = do
      print ("n" ++ show n ++ ", d" ++ show d ++ ", m" ++ show m)
      (measured, endtime) <- measure (nf p ( A.fromList A.Z [n]
                               , A.fromList A.Z [d]
                               , A.fromList A.Z [m])
                               ) 10
      print  (secs (measTime measured / 10))
