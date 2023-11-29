module Main where
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion
import Criterion.Main

import Quickhull

main :: IO ()
main = do
  inputs <- mapM load ["1M_rectangle_16384", "1M_circle_16384", "1M_quadratic_2147483648"]

  let quickhullCPU = CPU.runN quickhull
  let quickhullGPU = GPU.runN quickhull

  mapM_ (\input -> mapM_ (`testInput` input) [("CPU", quickhullCPU), ("GPU", quickhullGPU)]) inputs

  defaultMain [backend "CPU" quickhullCPU inputs, backend "GPU" quickhullGPU inputs]
  where
    backend name quickhull' inputs
      = bgroup name
      $ Prelude.map (testcase quickhull') inputs

    testcase quickhull' (name, points) =
      bench name $ nf quickhull' points

type Input = (String, A.Vector Point)

load :: String -> IO Input
load name = do
  putStrLn $ "Loading " ++ name
  content <- readFile $ "../input/" ++ name ++ ".dat"
  let list = map parseLine $ lines content
  list `seq` return (name, A.fromList (A.Z A.:. length list) list)
  where
    parseLine :: String -> Point
    parseLine line = case words line of
      [x, y] -> (read x, read y)
      _ -> error "Parse error"

testInput :: (String, A.Vector Point -> A.Vector Point) -> Input -> IO ()
testInput (backend, f) (inputName, inputData) = do
  putStrLn $ backend ++ "/" ++ inputName
  putStrLn $ take 80 $ show $ f inputData
  putStrLn ""
