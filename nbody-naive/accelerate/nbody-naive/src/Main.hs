module Main where
import NBody
import Data.Array.Accelerate (fromList, Z(..))
import Data.Array.Accelerate.LLVM.Native as CPU
import Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion.Measurement
import Criterion.Measurement.Types (Benchmarkable, Measured(measTime), nf)
import Control.Monad (replicateM)
import System.Environment (getArgs)
import System.IO (hPutStrLn, stderr)
import Numeric

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
          bench (nf (runN nbody (fromList Z [0.1]) (fromList Z [n])) (fromList Z [t]))
        "compiletime" -> do
          hPutStrLn stderr $ "Compilation time on " ++ backend
          time <- measTime . Prelude.fst <$> measure (nf runN nbody) 1
          print time
        "single" -> do
          hPutStrLn stderr $ "Single run " ++ backend ++ " " ++ show (n, t)
          let result = runN nbody (fromList Z [0.1]) (fromList Z [n]) (fromList Z [t])
          result `seq` return ()
        _ -> error "Unsupported mode"

    _ -> do
      hPutStrLn stderr "Usage: cabal run nbody-naive -- mode backend input"
      hPutStrLn stderr "mode: bench, compiletime or single"
      hPutStrLn stderr "backend: cpu or gpu"
      hPutStrLn stderr "input: n1000, n10000 or n100000"
      hPutStrLn stderr ""
      hPutStrLn stderr "Alternatively, run `sh run.sh` to run all measurements on all backends"

bench :: Benchmarkable -> IO ()
bench b = do
  -- Warm-up
  _ <- measure b 1
  go []
  where
    go :: [Double] -> IO ()
    go results
      | n >= 10 && e < 0.03 = do
        hPutStrLn stderr $ show (sum results / fromIntegral n) ++
          "s (" ++ show n ++ " runs, stderr " ++ showFFloat (Just 2) (e * 100) "%)"
        return ()
      | otherwise = do
        result <- measTime . Prelude.fst <$> measure b 1
        print result
        go (result : results)
      where
        n = length results
        -- Measure standard error of *throughput* (gigaflops), not of execution
        -- time, as we report throughput.
        e = stderrRatio (map (1/) results)

    -- stderr / mu = stddev / sqrt n / mu, where mu is average of measurements
    stderrRatio :: [Double] -> Double
    stderrRatio results = sqrt (sum [(x - mu) * (x - mu) | x <- results] / (n - 1)) / sqrt n / mu
      where
        n = fromIntegral (length results) :: Double
        mu = sum results / n
