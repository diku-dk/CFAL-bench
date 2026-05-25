{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
module Main where
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion.Measurement
import Criterion.Measurement.Types (Benchmarkable, Measured(measTime), nf)
import Control.Monad (replicateM)
import System.Environment (getArgs)
import System.IO (hPutStrLn, stderr)
import Numeric

import Quickhull
import qualified Data.ByteString as B
import qualified Data.ByteString.Internal as BI
import Data.Int
import Foreign.ForeignPtr (ForeignPtr, castForeignPtr)
import qualified Data.Array.Accelerate.IO.Foreign.ForeignPtr as A

main :: IO ()
main = do
  args <- getArgs

  case args of
    [mode, backend, inputName] -> do
      let runN = case backend of
            "cpu" -> CPU.runN
            "gpu" -> GPU.runN
            _ -> error "Unsupported backend"
    
      input <- load inputName

      case mode of
        "bench" -> do
          hPutStrLn stderr $ "Benchmark " ++ backend ++ " " ++ inputName
          bench (nf (runN quickhull) input)
        "compiletime" -> do
          hPutStrLn stderr $ "Compilation time on " ++ backend
          time <- measTime . Prelude.fst <$> measure (nf runN quickhull) 1
          print time
        "single" -> do
          hPutStrLn stderr $ "Single run " ++ backend ++ " " ++ inputName
          let result = runN quickhull input
          result `seq` return ()
        _ -> error "Unsupported mode"

    _ -> do
      hPutStrLn stderr "Usage: cabal run quickhull -- mode backend input"
      hPutStrLn stderr "mode: bench, compiletime or single"
      hPutStrLn stderr "backend: cpu or gpu"
      hPutStrLn stderr "input: 100M_rectangle, 100M_circle or 100M_quadratic"
      hPutStrLn stderr ""
      hPutStrLn stderr "Alternatively, run `sh run.sh` to run all measurements on all backends"

type Input = A.Vector Point

load :: String -> IO Input
load name = do
  hPutStrLn stderr $ "Loading " ++ name
  content <- B.readFile $ "../input/" ++ name ++ ".dat"
  let (fptrw8, nw8) = BI.toForeignPtr0 content
      arr = A.fromForeignPtrs (A.Z A.:. (nw8 `quot` 4)) (castForeignPtr fptrw8 :: ForeignPtr Int32) :: A.Array (A.Z A.:. Int) Int32
      res = A.fromFunction (A.Z A.:. (nw8 `quot` 8)) (\(A.Z A.:. ix) -> (fromIntegral $ A.indexArray arr (A.Z A.:. 2*ix), fromIntegral $ A.indexArray arr (A.Z A.:. 2*ix+1))) 
  return res


testInput :: (String, A.Vector Point -> A.Vector Point) -> (String, Input) -> IO ()
testInput (backend, f) (inputName, inputData) = do
  putStrLn $ backend ++ "/" ++ inputName
  putStrLn $ take 80 $ show $ f inputData
  putStrLn ""

chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk i xs = let (f, r) = splitAt i xs in f : chunk i r

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
        -- In other benchmarks we report throughput, but here we report
        -- execution time. Hence we also compute the standard error in terms of
        -- time.
        e = stderrRatio results

    -- stderr / mu = stddev / sqrt n / mu, where mu is average of measurements
    stderrRatio :: [Double] -> Double
    stderrRatio results = sqrt (sum [(x - mu) * (x - mu) | x <- results] / (n - 1)) / sqrt n / mu
      where
        n = fromIntegral (length results) :: Double
        mu = sum results / n
