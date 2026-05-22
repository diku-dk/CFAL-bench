{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX    as GPU

import Control.Concurrent (getNumCapabilities, threadDelay)
import Control.Exception (evaluate)
import Control.Monad (forM, forM_, when, replicateM)
import Criterion.Measurement
import Criterion.Measurement.Types (Measured(measTime), nf)
import Data.List (intercalate, nub)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe)
import Numeric (showFFloat)
import System.Environment (getArgs)
import System.IO (hFlush, stdout, hPutStrLn, stderr)
import System.Mem (performGC)

import Prelude hiding ((^))
import qualified Prelude ((^))

import qualified Naive as N
import qualified Flash_custom as C
import qualified Flash_alg1 as F
import Input


-- type-restricted version to avoid a bunch of pointless warnings
(^) :: Num a => a -> Int -> a
(^) = (Prelude.^)

-- (N, d, M), where M has been hand-picked to be optimal for our machine and implementation
benchmarkCases :: [(Int, Int, Int)]
benchmarkCases =
  [(16384, 64, 2^22), (32768, 64, 2^23), (8192, 128, 2^22), (16384, 128, 2^23)]

testCases :: [(Int, Int, [Int])]
testCases =
  [(512, 64, [1024, 16384, 32768, 65536, 131072])]

main :: IO ()
main = do
  args <- getArgs

  -- We use the 'custom' attention.
  -- Alternative implementations are naive (N.flashAttention) and
  -- alg1 (F.flashAttention).
  -- Note that custom attention is not actually 'flash' attention, but this
  -- was used during the development of the implementations of attention in
  -- CFAL, as we initially looked at the real flash attention.
  let alg = C.flashAttention

  case args of
    ["test"] -> mainTest

    [mode, backend, inputName] -> do
      let runN = case backend of
            "cpu" -> CPU.runN
            "gpu" -> GPU.runN
            _ -> error "Unsupported backend"

      let (d, n) = case inputName of
            "d64-N16384" -> (64, 16384)
            "d64-N32768" -> (64, 32768)
            "d128-N8192" -> (128, 8192)
            "d128-N16384" -> (128, 16384)
            _ -> error "Unsupported input"

      let input = CPU.runN mkInput $ ascalar (n, d)

      case mode of
        "bench" -> do
          ncpu <- getNumCapabilities
          -- Increase number of runs for configurations that have more variance
          let runs
                | backend == "gpu" = 10
                | ncpu == 1 = 15
                | otherwise = 50
          hPutStrLn stderr $ "Benchmark " ++ backend ++ " " ++ inputName
          times <- Prelude.map (measTime . Prelude.fst) Prelude.<$> replicateM (runs + 1) (measure (nf (runN alg) input) 1)
          mapM_ print $ tail times
        "compiletime" -> do
          hPutStrLn stderr $ "Compilation time on " ++ backend
          time <- measTime . Prelude.fst <$> measure (nf runN alg) 1
          print time
        "single" -> do
          hPutStrLn stderr $ "Single run " ++ backend ++ " " ++ inputName
          let result = runN alg input
          result `seq` return ()
        _ -> error "Unsupported mode"

    _ -> do
      hPutStrLn stderr "Usage: cabal run flashattention -- mode backend input"
      hPutStrLn stderr "Or:    cabal run flashattention -- test"
      hPutStrLn stderr "mode: bench, compiletime or single"
      hPutStrLn stderr "backend: cpu or gpu"
      hPutStrLn stderr "input: d64-N16384, d64-N32768, d128-N8192 or d128-N16384"
      hPutStrLn stderr ""
      hPutStrLn stderr "Alternatively, run `sh run.sh` to run all measurements on all backends"

type Input = (A.Matrix Float, A.Matrix Float, A.Matrix Float)

mainTest :: IO ()
mainTest = do
  let !cpu_naive = CPU.runN N.flashAttention
      !cpu_custom = CPU.runN C.flashAttention
      !cpu_alg1 = CPU.runN F.flashAttention
  let !mkInputFun = CPU.runN mkRandomInput
  let !similarFun = CPU.runN (\a b -> A.and $ A.flatten $ A.zipWith closeIsh a b)
                      where closeIsh x y = abs (x - y) A.< 1e-4
      similar a b = similarFun a b `A.indexArray` A.Z

  let adaptedBenchmarkCases = [(nN, d, [mM]) | (nN, d, mM) <- benchmarkCases]
  forM_ (testCases ++ adaptedBenchmarkCases) $ \(nN, d, mMs) -> do
    forM_ [1..5] $ \seed -> do
      putStrLn $ "Test N=" ++ show nN ++ " d=" ++ show d ++ " (seed " ++ show seed ++ ")"
      let !input = mkInputFun (ascalar seed) (ascalar (nN, d))
      let !out_naive = cpu_naive input
      let out_custom = cpu_custom input
      let outs_alg1 = map (cpu_alg1 input . ascalar) mMs
      forM_ ((out_custom, "custom") : [(o, "alg1 M=" ++ show mM) | (o, mM) <- zip outs_alg1 mMs]) $ \(out, descr) -> do
        putStrLn $ "  " ++ descr
        when (not (similar out_naive out)) $ do
          putStrLn $ "    Not similar: naive and " ++ descr
          putStrLn $ "    " ++ take 120 (replace '\n' ' ' (show out_naive))
          putStrLn $ "    " ++ take 120 (replace '\n' ' ' (show out))

-- Given N d M, check whether M is valid for the (N, d) pair as the tuning parameter for Flash_alg1.
validMparam :: Int -> Int -> Int -> Bool
validMparam nN d mM =
  let bc = (mM + 4 * d - 1) `div` (4 * d)
      br = min bc d
  in nN `mod` bc == 0 && nN `mod` br == 0

ascalar :: A.Elt a => a -> A.Scalar a
ascalar x = A.fromList A.Z [x]

replace :: Eq a => a -> a -> [a] -> [a]
replace needle repl = map (\x -> if x == needle then repl else x)
