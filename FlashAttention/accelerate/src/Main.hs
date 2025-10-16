{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE FlexibleContexts #-}
module Main where

import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX    as GPU

import Control.Exception (evaluate)
import Control.Monad (forM, forM_, when, replicateM)
import Criterion
import Criterion.Types (measTime)
import Criterion.Measurement (measure)
import Data.List (intercalate, nub)
import qualified Data.Map.Strict as Map
import Data.Maybe (fromMaybe)
import Numeric (showFFloat)
import System.Environment (getArgs)
import System.IO (hFlush, stdout)

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
  case args of
    [] -> mainBench (Left C.flashAttention) "custom"
    ["-bench", "naive"] -> mainBench (Left N.flashAttention) "naive"
    ["-bench", "custom"] -> mainBench (Left C.flashAttention) "custom"
    ["-bench", "alg1"] -> mainBench (Right F.flashAttention) "alg1"
    ["-test"] -> mainTest
    _ -> error "Arguments not understood, see 'main' src/Main.hs for options"

type Input = (A.Matrix Float, A.Matrix Float, A.Matrix Float)
mainBench :: Either (A.Acc Input                         -> A.Acc (A.Matrix Float))  -- algorithms that take no M (naive and custom)
                    (A.Acc Input -> A.Acc (A.Scalar Int) -> A.Acc (A.Matrix Float))  -- algorithms that take M (full flash attention)
          -> String
          -> IO ()
mainBench programE programName = do
  let !cpu = CPU.runN program
      !gpu = GPU.runN program
  let !cpuMkInput = CPU.runN mkInput
      !gpuMkInput = GPU.runN mkInput

  _ <- evaluate $ A.arraySize $ cpu (cpuMkInput (ascalar (512, 64))) (ascalar 8)  -- warmup
  tab1 <- forM benchmarkCases $ \inp ->
    benchSingle 20 ("CPU " ++ programName) cpu cpuMkInput inp

  _ <- evaluate $ A.arraySize $ gpu (gpuMkInput (ascalar (512, 64))) (ascalar 8)  -- warmup
  tab2 <- forM benchmarkCases $ \inp ->
    benchSingle 10 ("GPU " ++ programName) gpu gpuMkInput inp

  putStr $ printTable (tab1 <> tab2)
  where
    (program, printM) =
      case programE of
        Left f  -> (\input _  -> f input   , False)
        Right f -> (\input mM -> f input mM, True )

    benchSingle nruns descr fun mkInputFun (nN, d, mM) = do
      when (not (validMparam nN d mM)) $ error $ "Illegal parameters " ++ show (nN, d, mM) ++ "; ensure Br, Bc | N"

      putStr $ descr ++ " N=" ++ show nN ++ " d=" ++ show d ++ (if printM then " M=" ++ show mM else "") ++ ": "
      hFlush stdout
      let !input = mkInputFun (ascalar (nN, d))
      times <- replicateM nruns $ measTime . fst <$> measure (nf (uncurry fun) (input, ascalar mM)) 1

      let mean = sum times / fromIntegral nruns
          -- standard error, i.e. standard deviation of the mean estimate
          mean_stderr = sqrt (1 / (fromIntegral nruns - 1) * sum [(t - mean) ^ 2 | t <- times]) / sqrt (fromIntegral nruns)
      let flops = fromIntegral nN ^ 2 * (4.0 * fromIntegral d + 5.0)
      let gflopsPerSec = flops/1e9 / mean
      putStrLn (formatSecs mean ++ " ± " ++ formatSecs mean_stderr ++ " (" ++ showFFloat (Just 2) gflopsPerSec " Gflops/s) " ++ intercalate "," (map formatSecs times))

      return (descr, (nN, d), gflopsPerSec)

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

printTable :: [(String, (Int, Int), Double)] -> String
printTable items =
  let items' = [(impl, show inp, showFFloat (Just 2) val "") | (impl, inp, val) <- items]
      impls = nub [impl | (impl, _, _) <- items']
      inputs = nub [inp | (_, inp, _) <- items']
      implLen = maximum (map length impls)
      colValLen = [maximum [length val | (_, inp', val) <- items', inp == inp']
                  | inp <- inputs]
      colLen = zipWith max (map length inputs) colValLen
      mp = Map.fromList [((impl, inp), val) | (impl, inp, val) <- items']
      alignL w s = s ++ replicate (w - length s) ' '
      alignC w s = let n = w - length s
                   in replicate (n `div` 2) ' ' ++ s ++ replicate ((n + 1) `div` 2) ' '
      alignR w s = replicate (w - length s) ' ' ++ s
  in unlines (intercalate " " (alignC implLen "(N,d)" : zipWith alignC colLen inputs)
              : [intercalate " "
                   (alignL implLen impl : [alignC w (alignR valW (fromMaybe "" (Map.lookup (impl, inp) mp)))
                                          | (w, valW, inp) <- zip3 colLen colValLen inputs])
                | impl <- impls])

formatSecs :: Double -> String
formatSecs time = showFFloat (Just 4) time "s"

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
