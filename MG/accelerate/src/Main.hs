module Main where
import MG
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion.Measurement
import Criterion.Measurement.Types (Benchmarkable, Measured(measTime), nf)
import System.Environment (getArgs)
import System.IO (hPutStrLn, stderr)
import Numeric

main :: IO ()
main = do
  args <- getArgs

  case args of
    [mode, backend, inputName] -> do
      let runN = case backend of
            "cpu" -> CPU.runN
            "gpu" -> GPU.runN
            _ -> error "Unsupported backend"
      
      let (iter, n, weights) = case inputName of
            "A" -> (4, 256, weightsA)
            "B" -> (20, 256, weightsB)
            "C" -> (20, 512, weightsB)
            _ -> error "Unsupported input"

      let input = CPU.runN makeInput $ A.fromList A.Z [n]

      let f inp =
            mg n
              weights (A.use $ A.fromList A.Z [iter])
              inp
              (A.generate (A.Z_ A.::. A.constant n A.::. A.constant n A.::. A.constant n) $ const 0)

      case mode of
        "bench" -> do
          hPutStrLn stderr $ "Benchmark " ++ backend ++ " " ++ inputName
          bench (nf (runN f) input)
        "compiletime" -> do
          hPutStrLn stderr $ "Compilation time on " ++ backend ++ " " ++ inputName
          time <- measTime . Prelude.fst <$> measure (nf runN f) 1
          print time
        "single" -> do
          hPutStrLn stderr $ "Single run " ++ backend ++ " " ++ inputName
          let result = runN f input
          result `seq` return ()
        _ -> error "Unsupported mode"

    _ -> do
      hPutStrLn stderr "Usage: cabal run mg -- mode backend input"
      hPutStrLn stderr "mode: bench, compiletime or single"
      hPutStrLn stderr "backend: cpu or gpu"
      hPutStrLn stderr "input: A, B or C"
      hPutStrLn stderr ""
      hPutStrLn stderr "Alternatively, run `sh run.sh` to run all measurements on all backends"

weightsA, weightsB :: (Double, Double, Double, Double)
weightsA = (-3/8, 1/32, -1/64, 0)
weightsB = (-3/17, 1/33, -1/61, 0)

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
