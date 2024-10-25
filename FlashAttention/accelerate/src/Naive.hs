{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeOperators #-}

module Naive where

import Data.Array.Accelerate
import Prelude hiding (replicate, zipWith, map, sum)

-- (attempt at) direct port of 'naive.sac' to Accelerate. Untested.

flashAttention :: Acc (Matrix Float) -> Acc (Matrix Float) -> Acc (Matrix Float) -> Acc (Matrix Float)
flashAttention q k v =
  let kt = transpose k
      s = matmul q kt
      p = softmax s
  in matmul p v

softmax :: Acc (Matrix Float) -> Acc (Matrix Float)
softmax x = 
  let Z_ ::. cols ::. rows = shape x
      ex = map exp x
      s  = sum ex
      ss = replicate (Z_ ::. All_ ::. rows) s
  in  zipWith (/) ex ss

matmul :: Acc (Matrix Float) -> Acc (Matrix Float) -> Acc (Matrix Float)
matmul x y =
  case (shape x, shape y) of
    (Z_ ::. rows ::. _cols, Z_ ::. _rows ::. cols) ->
      fold1 (+) $ 
        transpose' $
          zipWith (*)
            (replicate (Z_ ::. All_ ::. All_ ::. cols) x)
            (replicate (Z_ ::. rows ::. All_ ::. All_) y)

transpose' :: (Shape sh, Elt a) => Acc (Array (sh :. Int:.Int) a) -> Acc (Array (sh :. Int:.Int) a)
transpose' x =
  let sh ::. a ::. b = shape x
  in backpermute (sh ::. b ::. a) (\(sh ::. b ::. a) -> sh ::. a ::. b) x

