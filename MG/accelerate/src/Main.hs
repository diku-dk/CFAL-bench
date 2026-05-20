module Main where
import MG
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion.Measurement
import Criterion.Measurement.Types (Measured(measTime), nf)
import Control.Monad (replicateM)
import System.Environment (getArgs)
import System.IO (hPutStrLn, stderr)

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
          times <- Prelude.map (measTime . Prelude.fst) Prelude.<$> replicateM 11 (measure (nf (runN f) input) 1)
          mapM_ print $ tail times
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
