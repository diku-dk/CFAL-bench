{-# LANGUAGE BangPatterns #-}
module Main where
import NBody
import Data.Array.Accelerate (fromList, Z(..))
import Data.Array.Accelerate.LLVM.Native as CPU
import Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion.Measurement
import Criterion.Measurement.Types (Measured(measTime), nf)
import Control.Monad (replicateM)
import System.Environment (getArgs)
import System.IO (hPutStrLn, stderr)

-- read input, run accelerate, benchmark
main :: IO ()
main = do
  args <- getArgs

  case args of
    [mode, backend, input] -> do
      let runN = case backend of
            "cpu" -> CPU.runN
            "gpu" -> GPU.runN
            _ -> error "Unsupported backend"
      let n, t :: Int
          (n, t) = case input of
            "n1000"   -> (1000, 100000)
            "n10000"  -> (10000,  1000)
            "n100000" -> (100000,   10)
            _ -> error "Unsupported input"

      case mode of
        "bench" -> do
          hPutStrLn stderr $ "Benchmark " ++ backend ++ " " ++ show (n, t)
          times <- Prelude.map (measTime . Prelude.fst) Prelude.<$> replicateM 11 (measure (nf (runN nbody (fromList Z [0.1]) (fromList Z [n])) (fromList Z [t])) 1)
          mapM_ print $ tail times -- First run is warm-up run
        "compiletime" -> do
          hPutStrLn stderr $ "Compilation time on " ++ backend
          times <- Prelude.map (measTime . Prelude.fst) Prelude.<$> replicateM 11 (measure (nf runN nbody) 1)
          mapM_ print $ tail times -- First run is warm-up run
        "single" -> do
          hPutStrLn stderr $ "Single run " ++ backend ++ " " ++ show (n, t)
          let result = runN nbody (fromList Z [0.1]) (fromList Z [n]) (fromList Z [t])
          result `seq` return ()

    _ -> do
      hPutStrLn stderr "Usage: cabal run nbody-naive -- mode backend input"
      hPutStrLn stderr "mode: bench, compiletime or single"
      hPutStrLn stderr "backend: cpu or gpu"
      hPutStrLn stderr "input: n1000, n10000 or n100000"
