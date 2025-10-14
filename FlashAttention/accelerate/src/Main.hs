{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE BangPatterns #-}
module Main where
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
-- import qualified Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion

import Flash_alg1
import Criterion.Types (measTime, Benchmark(Benchmark))
import Criterion.Measurement (measure, secs)

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
  -- print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^20])
  -- print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^21])
  -- print $ CPU.runN check (A.fromList A.Z [32768], A.fromList A.Z [64], A.fromList A.Z [2^22])
  let !cpu = CPU.runN totalProgram
  print $ A.arraySize $ cpu (A.fromList A.Z [512], A.fromList A.Z [64], A.fromList A.Z [16384])
  backend "CPU" cpu

  where
    backend name p
      = do
        putStrLn name
        mapM_ (testcase p)
          -- $ map (\(a,b)->(b,a)) 
          $ [(16384,64,2^(22 :: Int)),(32768,64,2^(23 :: Int)),(8192,128,2^(22 :: Int)),(16384,128,2^(23 :: Int))]
            --concatMap (\(d,n) -> [(n,d,m) | m <- [2^22,2^23]]) 
            --[(64,16384),(64,32768),(128,8192),(128,16384)]
      -- $ (,,) <$> [512] --, 1024, 2048, 4096, 8192, 16384]
      --        <*> [64] --, 128]
      --        <*> [8] --, 64, 512, 2048, 8192, 16384]
    testcase p (n,d,m) = do
      checkParams n d m
      putStrLn ("n" ++ show n ++ ", d" ++ show d ++ ", m" ++ show m)
      (measured, _) <- measure (nf p ( A.fromList A.Z [n]
                               , A.fromList A.Z [d]
                               , A.fromList A.Z [m])
                               ) 10
      let mean = measTime measured / 10
      putStrLn  (secs mean)
      let flops = (4.0 * fromIntegral d + 5.0) * fromIntegral n * fromIntegral n
      let gflopsPerSec = flops / mean / 1e9
      putStrLn (show gflopsPerSec ++ " Gflops/s")
      putStrLn ""

checkParams :: Int -> Int -> Int -> IO ()
checkParams n d m
  | bc <- (m + 4 * d - 1) `div` (4 * d)
  , br <- min bc d
  , (n `mod` br /= 0) || (n `mod` bc /= 0) = error "Illegal parameters. Make sure Br, Bc | N"
  | otherwise = return ()
