{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE TypeOperators #-}

module Naive where

import Data.Array.Accelerate
import Prelude hiding (replicate, zipWith, map, sum)
--import Data.Array.Accelerate.Numeric.LinearAlgebra

-- (attempt at) direct port of 'naive.sac' to Accelerate. Untested.


flashAttention :: Acc (Matrix Float) -> Acc (Matrix Float) -> Acc (Matrix Float) -> Acc (Matrix Float)
flashAttention q k v =
  let --kt = transpose k
      --s = matmul q kt
      s = matmulT q k
      p = softmax s
  in matmul p v

softmax :: Acc (Matrix Float) -> Acc (Matrix Float)
softmax x = 
  let Z_ ::. _cols ::. rows = shape x
      ex = map exp x
      s  = sum ex
      ss = replicate (Z_ ::. All_ ::. rows) s
  in  zipWith (/) ex ss

-- Given arrays of size (sh, m, k) and (sh, n, k),
-- Returns an array of size (sh, m, n)
matmulT :: Shape sh => Acc (Array (sh :. Int :. Int) Float) -> Acc (Array (sh :. Int :. Int) Float) -> Acc (Array (sh :. Int :. Int) Float)
matmulT a b = fold (+) 0 $ zipWith (*)
    -- Replicate a and b to arrays of size (sh, m, n, k)
    (replicate (Any_ ::. All_ ::. n ::. All_) a)
    (replicate (Any_ ::. m ::. All_ ::. All_) b)
  where
    (_ ::. m ::. _) = shape a
    (_ ::. n ::. _) = shape b

-- Given arrays of size (sh, m, k) and (sh, k, n),
-- Returns an array of size (sh, m, n)
matmul :: Shape sh => Acc (Array (sh :. Int :. Int) Float) -> Acc (Array (sh :. Int :. Int) Float) -> Acc (Array (sh :. Int :. Int) Float)
matmul a b = matmulT a (transpose' b)

transpose' :: (Shape sh, Elt a) => Acc (Array (sh :. Int:.Int) a) -> Acc (Array (sh :. Int:.Int) a)
transpose' x =
  let sh ::. m ::. n = shape x
  in backpermute (sh ::. n ::. m) (\(idx ::. i ::. j) -> idx ::. j ::. i) x
