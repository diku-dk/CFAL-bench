{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use camelCase" #-}
{-# LANGUAGE TypeOperators #-}
module Flash_custom where

import Data.Array.Accelerate hiding (encodeFloat,(^))
import Prelude hiding (replicate, zipWith, zipWith3, map, sum, min, Ord(..), maximum)


-- N = m*d
-- Q,K,V: N * d
-- capitalized variable names are written doubled, e.g. N is rendered nN, S is rendered sS.
flashAttention :: Acc (Matrix Float, Matrix Float, Matrix Float) -> Acc (Matrix Float)
flashAttention (T3 q k v) =
  let Z_ ::. nN ::. d = shape q
      m = nN `quot` d
      sS = matmulT (reshape (I3 m d d) q) (replicate (I3 m All_ All_) k)  -- m * d * N
      stabilized = zipWith (-) sS (replicate (I3 All_ All_ nN) (maximum sS))  -- m * d * N
      exped = map exp stabilized  -- m * d * N
      scaled = zipWith (*) exped (replicate (I3 All_ All_ nN) (map recip (sum exped)))  -- m * d * N
      result = matmulT scaled (replicate (I3 m All_ All_) $ compute $ transpose v)  -- m * (d*N @ N*d = d*d): m * d * d
  in reshape (Z_ ::. nN ::. d) result  -- N * d

-- Given arrays of size (sh, m, k) and (sh, n, k),
-- Returns an array of size (sh, m, n)
matmulT :: Shape sh => Acc (Array (sh :. Int :. Int) Float) -> Acc (Array (sh :. Int :. Int) Float) -> Acc (Array (sh :. Int :. Int) Float)
matmulT a b =
  fold (+) 0 $ zipWith (*)
    -- Replicate a and b to arrays of size (sh, m, n, k)
    (replicate (Any_ ::. All_ ::. n ::. All_) a)
    (replicate (Any_ ::. m ::. All_ ::. All_) b)
  where
    (_ ::. m ::. _) = shape a
    (_ ::. n ::. _) = shape b

transpose' :: (Shape sh, Elt a) => Acc (Array (sh :. Int :. Int) a) -> Acc (Array (sh :. Int :. Int) a)
transpose' x =
  let sh ::. m ::. n = shape x
  in backpermute (sh ::. n ::. m) (\(idx ::. i ::. j) -> idx ::. j ::. i) x
