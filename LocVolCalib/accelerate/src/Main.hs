module Main where

import LocVolCalib

import Criterion
import Data.Array.Accelerate
import qualified Data.Array.Accelerate.Interpreter as Interp
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX as GPU
import Prelude hiding (sum)

-- currently does NOT work: returns only NaNs. Not sure why, it seems like a faithful translation from Futhark.
main :: IO ()
main = 
  defaultMain [backend "CPU" CPU.runN, backend "GPU" GPU.runN]

    where
      backend name run = bgroup name $ map benchrun run [small, medium, large]
      benchrun run (name, input) = bench name $ nf (run main') input

      small = ("small", T9
                        (constant 16)
                        (constant 32)
                        (constant 256)
                        (constant 256)
                        (constant 0.03)
                        (constant 5.0)
                        (constant 0.2)
                        (constant 0.6)
                        (constant 0.5))
      medium = ("medium", T9 
                        (constant 128 )
                        (constant 256 )
                        (constant 32  )
                        (constant 64  )
                        (constant 0.03)
                        (constant 5.0 )
                        (constant 0.2 )
                        (constant 0.6 )
                        (constant 0.5 ))
      large = ("large", T9 
                        (constant 256 )
                        (constant 256 )
                        (constant 256 )
                        (constant 64  )
                        (constant 0.03)
                        (constant 5.0 )
                        (constant 0.2 )
                        (constant 0.6 )
                        (constant 0.5 ))

  print $ CPU.runN $ -- const (sum $ generate (Z_ ::. (constant 40 :: Exp Int)) (const (constant 40 :: Exp Int))) $
    main' $ T9 
        -- small
      -- (constant 16)
      -- (constant 32)
      -- (constant 256)
      -- (constant 256)
      -- (constant 0.03)
      -- (constant 5.0)
      -- (constant 0.2)
      -- (constant 0.6)
      -- (constant 0.5)

        -- medium
      (constant 128 )
      (constant 256 )
      (constant 32  )
      (constant 64  )
      (constant 0.03)
      (constant 5.0 )
      (constant 0.2 )
      (constant 0.6 )
      (constant 0.5 )
