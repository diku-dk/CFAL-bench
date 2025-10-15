{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use camelCase" #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ViewPatterns #-}
module Flash_alg1 where

import Data.Array.Accelerate hiding (encodeFloat,(^))
import Prelude hiding (replicate, zipWith, zipWith3, map, sum, min, Ord(..), maximum)


flashAttention :: Acc (Matrix Float, Matrix Float, Matrix Float) -> Acc (Scalar Int) -> Acc (Matrix Float)
flashAttention (T3 q k v) (the -> m') =
  let Z_ ::. n ::. d = shape q
      bc = ceildiv m' (4*d)
      br = min bc d

      qb = reshape (Z_ ::. n `div` br ::. br ::. d) q
      kb = reshape (Z_ ::. n `div` bc ::. bc ::. d) k
      vb = reshape (Z_ ::. n `div` bc ::. bc ::. d) v

      o = fill (Z_ ::. n `div` br ::. br ::. d) 0
      m = fill (Z_ ::. n `div` br ::. br)       (negate real_max)
      l = fill (Z_ ::. n `div` br ::. br)       0

      max_j = n `div` bc
      T3 result _ _ = afst $ awhile (map (< max_j) . asnd)
             (\(T2 state j) -> T2 (step state qb kb vb j) (map (+1) j))
             (T2 (T3 o m l) $ unit 0)
  in reshape (Z_ ::. n ::. d) result

real_max :: Exp Float
real_max = constant $ encodeFloat (2^(24 :: Int) - 1) (127-23)--3.40282346638528859811704183484516925e+38, largest non-infinity float

type State = Acc (Array DIM3 Float,Matrix Float,Matrix Float)

step :: State -> Acc (Array DIM3 Float) -> Acc (Array DIM3 Float) -> Acc (Array DIM3 Float) -> Acc (Scalar Int) -> State
step (T3 o m l) qb kb vb j =
  let Z_ ::. _ ::. bc ::. d = shape kb
      Z_ ::. nbr ::. _br ::. _d = shape o
      kbj = replicate (Z_ ::. nbr ::. All_ ::. All_) $ slice kb $ Z_ ::. the j ::. All_ ::. All_
      vbj = replicate (Z_ ::. nbr ::. All_ ::. All_) $ slice vb $ Z_ ::. the j ::. All_ ::. All_
      T3 pj1 mj lj = exp_e $ matmulT qb kbj
      mnew = zipWith max m mj
      lnew = zipWith5 (\m_ mnew_ l_ mj_ lj_ -> exp (m_-mnew_) * l_ + exp (mj_ - mnew_) * lj_)
                        m  mnew  l  mj  lj
      o' = zipWith4 (\l_ m_ mnew_ o_ -> l_ * exp (m_ - mnew_) * o_)
            (replicate (Any_ ::. d) l)
            (replicate (Any_ ::. d) m)
            (replicate (Any_ ::. d) mnew)
            o
      pj2 = zipWith3 (\mj' mnew' pj1' -> exp (mj' - mnew')*pj1')
              (replicate (Any_ ::. bc) mj) 
              (replicate (Any_ ::. bc) mnew) 
              pj1
      o'' = zipWith (+) o' $ matmul pj2 vbj
      o''' = zipWith (/) o'' (replicate (Any_ ::. d) lnew)
  in T3 o''' mnew lnew

ceildiv :: Exp Int -> Exp Int -> Exp Int
ceildiv a b = (a + b - 1) `div` b

exp_e :: Shape sh => Acc (Array (sh :. Int) Float) -> Acc (Array (sh :. Int) Float, Array sh Float, Array sh Float)
exp_e x =
  let _ ::. sz = shape x
      thismax = maximum x
      fx = zipWith (\t x' -> exp(x' - t)) (replicate (Any_ ::. sz) thismax) x
  in  T3 fx thismax (sum fx)

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
matmul a b = matmulT (compute a) (compute $ transpose' b)

transpose' :: (Shape sh, Elt a) => Acc (Array (sh :. Int :. Int) a) -> Acc (Array (sh :. Int :. Int) a)
transpose' x =
  let sh ::. m ::. n = shape x
  in backpermute (sh ::. n ::. m) (\(idx ::. i ::. j) -> idx ::. j ::. i) x
