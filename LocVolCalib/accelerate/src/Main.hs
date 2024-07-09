{-# LANGUAGE PatternSynonyms #-}
module Main where

import LocVolCalib

import Criterion.Main
import Data.Array.Accelerate hiding (map)
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX as GPU
import Prelude hiding (sum)

-- currently does NOT work: returns only NaNs. Not sure why, it seems like a faithful translation from Futhark.
main :: IO ()
main = 
  defaultMain [backend "CPU" CPU.runN, backend "GPU" GPU.runN]
    where
      backend name run = bgroup name $ map (benchrun run) [small, medium, large]
      benchrun run (name, input) = bench name $ nf (run main') (fromList Z $ pure input)

      small = ("small", (16
                        ,32
                        ,256
                        ,256
                        ,0.03
                        ,5.0
                        ,0.2
                        ,0.6
                        ,0.5))
      medium = ("medium",(128 
                         ,256 
                         ,32  
                         ,64  
                         ,0.03
                         ,5.0 
                         ,0.2 
                         ,0.6 
                         ,0.5 ))
      large = ("large", (256 
                        ,256 
                        ,256 
                        ,64  
                        ,0.03
                        ,5.0 
                        ,0.2 
                        ,0.6 
                        ,0.5 ))

