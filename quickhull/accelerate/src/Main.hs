{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
module Main where
import qualified Data.Array.Accelerate as A
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion
import Criterion.Main

import Quickhull
import qualified Data.ByteString as B
import qualified Data.ByteString.Internal as BI
import GHC.IsList
import Data.Int
import Foreign.ForeignPtr (ForeignPtr, castForeignPtr)
import qualified Data.Array.Accelerate.IO.Foreign.ForeignPtr as A

main :: IO ()
main = do
  inputs <- mapM load ["1M_rectangle", "1M_circle", "1M_quadratic", "100M_rectangle", "100M_circle", "100M_quadratic"]

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
  content <- B.readFile $ "../input/" ++ name ++ ".dat"
  let (fptrw8, nw8) = BI.toForeignPtr0 content
      arr = A.fromForeignPtrs (A.Z A.:. (nw8 `quot` 4)) (castForeignPtr fptrw8 :: ForeignPtr Int32) :: A.Array (A.Z A.:. Int) Int32
      res = A.fromFunction (A.Z A.:. (nw8 `quot` 8)) (\(A.Z A.:. ix) -> (fromIntegral $ A.indexArray arr (A.Z A.:. 2*ix), fromIntegral $ A.indexArray arr (A.Z A.:. 2*ix+1))) 
  return (name, res)


testInput :: (String, A.Vector Point -> A.Vector Point) -> Input -> IO ()
testInput (backend, f) (inputName, inputData) = do
  putStrLn $ backend ++ "/" ++ inputName
  putStrLn $ take 80 $ show $ f inputData
  putStrLn ""

chunk :: Int -> [a] -> [[a]]
chunk _ [] = []
chunk i xs = let (f, r) = splitAt i xs in f : chunk i r
