{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE RebindableSyntax #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module LocVolCalib where

import Prelude (id)
import Data.Array.Accelerate
import Data.Array.Accelerate.Unsafe (undef)
import Data.Array.Accelerate.Control.Lens

type Two a = (a,a)
type Three a = (a,a,a)

-- WIP: blind translation from futhark

initGrid :: Exp (Float, Float, Float, Float, Int, Int, Int)
         -> Acc (Scalar Int32, Scalar Int32, Vector Float, Vector Float, Vector Float)
initGrid (T7 s0 alpha nu t numX numY numT) =
  let logAlpha = log alpha
      myTimeline = generate (I1 numT) (\(I1 i) -> t * toFloating i / (toFloating numT - 1.0))
      stdX = 20.0 * alpha * s0 * sqrt t
      stdY = 10.0 * nu         * sqrt t
      dx = stdX / toFloating numX
      dy = stdY / toFloating numY
      myXindex = round (s0 / dx) :: Exp Int32
      myYindex = fromIntegral numY `div` 2
      myX = generate (I1 numX) (\(I1 i) -> toFloating i * dx - toFloating myXindex * dx + s0)
      myY = generate (I1 numY) (\(I1 i) -> toFloating i * dy - toFloating myYindex * dy + logAlpha)
  in T5 (unit myXindex) (unit myYindex) myX myY myTimeline

-- TODO: individually match on less and more, and just fromJust in all known cases
-- probably makes no difference: we separately generate code for the border and interior,
-- and in both cases LLVM should see that only one branch ever gets taken
initOperator :: Acc (Vector Float) -> Acc (Vector (Two (Three Float)))
initOperator x = stencil
  (\(a,b,c) -> match $ T3 a b c & \case
    T3 (Just_ less) (Just_ here) (Just_ more) ->
      let dxl = here - less
          dxu = more - here
      in T2 (T3 (-dxu/dxl/(dxl+dxu)) ((dxu/dxl - dxl/dxu)/(dxl+dxu)) (dxl/dxu/(dxl+dxu)))
            (T3 (2/dxl/(dxl+dxu)   ) (-2*(1/dxl + 1/dxu)/(dxl+dxu) ) (2/dxu/(dxl+dxu)))
    T3 Nothing_ (Just_ here) (Just_ more) -> let dxu = more - here in T2 (T3 0 (-1/dxu) (1/dxu)) (T3 0 0 0)
    T3 (Just_ less) (Just_ here) Nothing_ -> let dxl = here - less in T2 (T3 (-1/dxl) (1/dxl) 0) (T3 0 0 0)
    _ -> undef -- should never be hit at runtime,but the compiler needs this case to exist
  )
  (function $ const Nothing_)
  (map Just_ x)

setPayoff :: Acc (Scalar Float, Vector Float, Vector Float) -> Acc (Matrix Float)
setPayoff (T3 strike myX myY) =
  let I1 numX = shape myX
      I1 numY = shape myY
  in replicate (Z_ ::. numY ::. All_) $ map (\xi -> max (xi - the strike) 0) myX

updateParams :: Acc (Vector Float, Vector Float) -> Exp (Three Float) -> Acc (Matrix Float, Matrix Float, Matrix Float, Matrix Float)
updateParams (T2 myX myY) (T3 tnow beta nu) = T4 myMuX myVarX myMuY myVarY
  where
    I1 numX = shape myX
    I1 numY = shape myY
    myMuY  = generate (Z_ ::. numX ::. numY) (const 0)
    myVarY = generate (Z_ ::. numX ::. numY) (const $ nu*nu)
    myMuX  = generate (Z_ ::. numY ::. numX) (const 0)
    myVarX = zipWith (\y x -> exp (2*(beta * log x + y - 0.5*nu*nu*tnow)))
              (replicate (Z_ ::. All_ ::. numX) myY)
              (replicate (Z_ ::. numY ::. All_) myX)

-- unused: in Accelerate, you can't `map` this function over matrices. Instead, see tridagParSh
tridagPar :: Acc (Vector Float, Vector Float, Vector Float, Vector Float) -> Acc (Vector Float)
tridagPar (T4 a b c y) =
  let I1 n = shape a
      -- recurrence 1
      b0 = b ! (Z_ ::. 0)
      mats = generate (I1 n) (\i@(I1 i')-> if 0 < i'
                                 then T4 (b ! i) (negate $ (a ! i) *(c ! I1 (i'-1))) 1 0
                                 else T4 1 0 0 1)
      scmt = scanl (\(T4 a0 a1 a2 a3) (T4 b0 b1 b2 b3) ->
                     let value = 1/(a0*b0)
                     in T4 ((b0*a0 + b1*a2)*value)
                           ((b0*a1 + b1*a3)*value)
                           ((b2*a0 + b3*a2)*value)
                           ((b2*a1 + b3*a3)*value))
                  (T4 1 0 0 1) mats
      b'    = map (\(T4 t0 t1 t2 t3) -> (t0*b0 + t1) / (t2*b0 + t3)) scmt
      -- recurrence 2
      y0   = y ! (Z_::.0)
      lfuns= generate (I1 n) (\i@(I1 i') ->
                   if 0 < i'
                   then T2 (y!i) (0.0-(a!i)/(b'!I1 (i'-1)))
                   else T2 0 1)
      cfuns= scanl (\(T2 a0 a1) (T2 b0 b1) -> T2 (b0 + b1*a0) (a1*b1))
                (T2 0 1) lfuns
      y'    = map (\(T2 a b)  -> a + b*y0) cfuns
      -- recurrence 3
      yn   = y' !! (n-1) / b' !! (n-1)
      lfuns' = generate (I1 n) (\(I1 k)  ->
                    let i = n-k-1
                    in  if   0 < k
                        then T2 (y' !! i / b' !! i) (0.0-c !! i / b' !! i)
                        else T2 0 1)
      cfuns' = scanl (\(T2 a0 a1) (T2 b0 b1) -> T2 (b0 + b1*a0) (a1*b1))
                  (T2 0 1) lfuns'
      y''    = map (\(T2 a b) -> a + b*yn) cfuns'
      y'''    = reverse y''
  in y'''
-- 'vectorized' version of the above
tridagParSh :: Shape sh => Acc (Array (sh:.Int) Float, Array (sh:.Int) Float, Array (sh:.Int) Float, Array (sh:.Int) Float) -> Acc (Array (sh:.Int) Float)
tridagParSh (T4 a b c y) =
  let sh@(_ ::. n) = shape a
      -- recurrence 1
      b0 = slice b (Any_ ::. (0 :: Exp Int))
      mats = generate sh (\i@(ix ::. i')-> if 0 < i'
                                 then T4 (b ! i) (negate $ (a ! i) *(c ! (ix ::. i'-1))) 1 0
                                 else T4 1 0 0 1)
      scmt = scanl (\(T4 a0 a1 a2 a3) (T4 b0 b1 b2 b3) ->
                     let value = 1/(a0*b0)
                     in T4 ((b0*a0 + b1*a2)*value)
                           ((b0*a1 + b1*a3)*value)
                           ((b2*a0 + b3*a2)*value)
                           ((b2*a1 + b3*a3)*value))
                  (T4 1 0 0 1) mats
      b'    = zipWith (\(T4 t0 t1 t2 t3) b0' -> (t0*b0' + t1) / (t2*b0' + t3)) scmt $ replicate (Any_ ::. n) b0
      -- recurrence 2
      y0   = slice y (Any_ ::. (0 :: Exp Int))
      lfuns= generate sh (\i@(ix ::. i') ->
                   if 0 < i'
                   then T2 (y!i) (0.0-(a!i)/(b'!(ix ::. i'-1)))
                   else T2 0 1)
      cfuns= scanl (\(T2 a0 a1) (T2 b0 b1) -> T2 (b0 + b1*a0) (a1*b1))
                (T2 0 1) lfuns
      y'    = zipWith (\(T2 a b) y0'  -> a + b*y0') cfuns $ replicate (Any_ ::. n) y0
      -- recurrence 3
      yn   = zipWith (/) (slice y' (Any_ ::. (n-1 :: Exp Int))) (slice b' (Any_ ::. (n-1 :: Exp Int)))
      lfuns' = generate sh (\(ix ::. k)  ->
                    let i = n-k-1
                    in  if   0 < k
                        then T2 (y' ! (ix ::. i) / b' ! (ix ::. i)) (0.0-c ! (ix ::. i) / b' ! (ix ::. i))
                        else T2 0 1)
      cfuns' = scanl (\(T2 a0 a1) (T2 b0 b1) -> T2 (b0 + b1*a0) (a1*b1))
                  (T2 0 1) lfuns'
      y''    = zipWith (\(T2 a b) yn' -> a + b*yn') cfuns' $ replicate (Any_ ::. n) yn
      y'''    = reverseOn _1 y''
  in y'''

explicitMethod :: Acc (Vector (Three Float), Vector (Three Float), Matrix Float, Matrix Float, Matrix Float)
               -> Acc (Matrix Float)
explicitMethod (T5 myD myDD myMu myVar result) =
  let I2 n m = shape myMu
  in zipWith5
      (\(T3 dx0 dx1 dx2) (T3 dxx0 dxx1 dxx2) mu var (T3 jprev j jnext) ->
        let c1 = (mu*dx0 + 0.5*var*dxx0) * jprev
            c3 = (mu*dx2 + 0.5*var*dxx2) * j
            c2 = (mu*dx1 + 0.5*var*dxx1) * jnext
        in  c1 + c2 + c3)
      (replicate (Z_ ::. n ::. All_) myD )
      (replicate (Z_ ::. n ::. All_) myDD)
      myMu
      myVar
      -- Currently using a 3x3 stencil and ignoring the top and bottom rows, 
      -- because Accelerate doesn't export a 1x3 stencil. 
      -- Hopefully the other 6 array reads get optimised away.
      (stencil @_ @(Three (Three (Exp Float))) (\(_,(a,b,c),_) -> T3 a b c) (function $ const 0) result)

implicitMethod :: Acc (Vector (Three Float), Vector (Three Float), Matrix Float, Matrix Float, Matrix Float) -> Exp Float -> Acc (Matrix Float)
implicitMethod (T5 myD myDD myMu myVar u) dtInv =
  let Z_ ::. n ::. m = shape myMu
      (a, b, c) = unzip3 $ zipWith4 (\mu var (T3 d0 d1 d2) (T3 dd0 dd1 dd2) -> T3
                                    (negate (0.5*(mu*d0 + 0.5*var*dd0)))
                                    (dtInv - 0.5*(mu*d1 + 0.5*var*dd1))
                                    (negate (0.5*(mu*d2 + 0.5*var*dd2))))
                                  myMu
                                  myVar
                                  (replicate (Z_ ::. n ::. All_) myD)
                                  (replicate (Z_ ::. n ::. All_) myDD)
  in tridagParSh $ T4 a b c u

rollback = undefined

value = undefined

main = undefined

