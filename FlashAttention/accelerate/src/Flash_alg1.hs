{-# OPTIONS_GHC -Wno-unrecognised-pragmas #-}
{-# HLINT ignore "Use camelCase" #-}
{-# LANGUAGE TypeOperators #-}
module Flash_alg1 where

import Data.Array.Accelerate
import Prelude hiding (replicate, zipWith, zipWith3, map, sum, min, Ord(..), maximum)

-- wip: port of flash_alg1.sac to accelerate

-- m is for blocking
flashAttention :: Acc (Matrix Float) -> Acc (Matrix Float) -> Acc (Matrix Float) -> Exp Int -> Acc (Matrix Float)
flashAttention q k v m' =
  let Z_ ::. n ::. d = shape q
      bc = ceildiv m' (4*d)
      br = min bc d

      qb = reshape (Z_ ::. n / br ::. br ::. d) q
      kb = reshape (Z_ ::. n / bc ::. bc ::. d) k
      vb = reshape (Z_ ::. n / bc ::. bc ::. d) v

      o = fill (Z_ ::. n / br ::. br ::. d) 0
      m = fill (Z_ ::. n / br ::. br)       (negate real_max)
      l = fill (Z_ ::. n / br ::. br)       0

      max_j = n / bc
      (T3 result _ _) = afst $ awhile (map (< max_j) . asnd)
             (\(T2 state j) -> T2 (step state qb kb vb j) (j+1))
             (T2 (T3 o m l) 0)
  in reshape (Z_ ::. n ::. d) result

real_max = 3.40282346638528859811704183484516925e+38

type State = Acc (Array DIM3 Float,Matrix Float,Matrix Float)

step :: State -> Acc (Array DIM3 Float) -> Acc (Array DIM3 Float) -> Acc (Array DIM3 Float) -> Acc (Scalar Int) -> State
step (T3 o m l) qb kb vb j =
  let Z_ ::. nbc ::. bc ::. d = shape kb
      kb' = replicate (Z_ ::. nbc ::. All_ ::. All_) $ slice kb $ Z_ ::. the j ::. All_ ::. All_
      T3 pj1 mj lj = exp_e $ matmulT qb kb'
      mnew = zipWith max m mj
      lnew = zipWith5 (\m_ mnew_ l_ mj_ lj_ -> exp (m_-mnew_) * l_ + exp (mj_ - mnew_) * lj_)
                        m  mnew  l  mj  lj
      o' = zipWith4 (\l_ m_ mnew_ o_ -> l_ * exp (m_ - mnew_) * o_)
            (replicate _ l)
            (replicate _ m)
            (replicate _ mnew)
            o
      pj2 = zipWith3 (\mj' mnew' pj1' -> exp (mj' - mnew')*pj1')
              (replicate (Z_ ::. All_ ::. All_ ::. d) mj) 
              (replicate (Z_ ::. All_ ::. All_ ::. d) mnew) 
              pj1
      o'' = undefined o' pj2 vbj
      o''' = zipWith (/) o (replicate _ lnew)
  in T3 o''' mnew lnew

ceildiv :: Exp Int -> Exp Int -> Exp Int
ceildiv a b = (a + b - 1) / b

exp_e :: Shape sh => Acc (Array (sh :. Int) Float) -> Acc (Array (sh :. Int) Float, Array sh Float, Array sh Float)
exp_e x =
  let _ ::. sz = shape x
      thismax = maximum x
      fx = zipWith (\t x' -> exp(x' - t)) (replicate (Any_ ::. sz) thismax) x
  in  T3 fx thismax (sum fx)

matmulT a b = matmul a $ transpose' b

matmul :: Shape sh => Acc (Array (sh :. Int :. Int) Float) -> Acc (Array (sh :. Int :. Int) Float) -> Acc (Array (sh :. Int :. Int) Float)
matmul x y =
  case (shape x, shape y) of
    (shx ::. rows ::. _cols, shy ::. _rows ::. cols) ->
      fold1 (+) $ 
        transpose' $
          zipWith (*)
            (replicate (Any_ ::. All_ ::. All_ ::. cols) x)
            (replicate (Any_ ::. rows ::. All_ ::. All_) y)

transpose' :: (Shape sh, Elt a) => Acc (Array (sh :. Int:.Int) a) -> Acc (Array (sh :. Int:.Int) a)
transpose' x =
  let sh ::. a ::. b = shape x
  in backpermute (sh ::. b ::. a) (\(sh ::. b ::. a) -> sh ::. a ::. b) x

