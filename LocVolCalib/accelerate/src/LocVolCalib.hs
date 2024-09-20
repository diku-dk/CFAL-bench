{-# LANGUAGE BlockArguments #-}
{-# LANGUAGE LambdaCase #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RebindableSyntax #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE ViewPatterns #-}

module LocVolCalib where

import Data.Array.Accelerate
import Data.Array.Accelerate.Unsafe (undef)
import Data.Array.Accelerate.Control.Lens

type Two a = (a,a)
type Three a = (a,a,a)
type Input = (Int, Int, Int, Int, Double, Double, Double, Double, Double)


initGrid :: Exp (Double, Double, Double, Double, Int, Int, Int)
         -> (Exp (Int, Int), Acc (Vector Double, Vector Double, Vector Double))
initGrid (T7 s0 alpha nu t numX numY numT) =
  let logAlpha = log alpha
      myTimeline = generate (I1 numT) (\(I1 i) -> t * toFloating i / (toFloating numT - 1.0))
      stdX = 20.0 * alpha * s0 * sqrt t
      stdY = 10.0 * nu         * sqrt t
      dx = stdX / toFloating numX
      dy = stdY / toFloating numY
      myXindex = floor (s0 / dx)
      myYindex = numY `div` 2
      -- redundancy fix 1
      myX = generate (I1 numX) (\(I1 i0) -> let i = toFloating i0 in i * log(i+1) * dx - toFloating myXindex * dx + s0)
      myY = generate (I1 numY) (\(I1 i0) -> let i = toFloating i0 in i * log(i+1) * dy - toFloating myYindex * dy + logAlpha)
  in (T2 myXindex myYindex, T3 myX myY myTimeline)

-- TODO: individually match on less and more, and just fromJust in all known cases
-- probably makes no difference: we separately generate code for the border and interior,
-- and in both cases LLVM should see that only one branch ever gets taken
initOperator :: Acc (Vector Double) -> Acc (Vector (Two (Three Double)))
initOperator x = stencil
  (\(a,b,c) -> T3 a b c & match (\case
    T3 (Just_ less) (Just_ here) (Just_ more) ->
      let dxl = here - less
          dxu = more - here
      in T2 (T3 (-dxu/dxl/(dxl+dxu)) ((dxu/dxl - dxl/dxu)/(dxl+dxu)) (dxl/dxu/(dxl+dxu)))
            (T3 (2/dxl/(dxl+dxu)   ) (-2*(1/dxl + 1/dxu)/(dxl+dxu) ) (2/dxu/(dxl+dxu)))
    T3 Nothing_ (Just_ here) (Just_ more) -> let dxu = more - here in T2 (T3 0 (-1/dxu) (1/dxu)) (T3 0 0 0)
    T3 (Just_ less) (Just_ here) Nothing_ -> let dxl = here - less in T2 (T3 (-1/dxl) (1/dxl) 0) (T3 0 0 0)
    _ -> undef -- should never be hit at runtime,but the compiler needs this case to exist
  ))
  (function $ const Nothing_)
  (map Just_ x)

setPayoff :: Acc (Vector Double, Vector Double, Vector Double) -> Acc (Array DIM3 Double)
setPayoff (T3 strikes myX myY) =
  let I1 numX = shape myX
      I1 numY = shape myY
      I1 numS = shape strikes
  in zipWith (\xi strike -> max (xi - strike) 0)
              (replicate (Z_ ::. numS ::. numY ::. All_) myX)
              (replicate (Z_ ::. All_ ::. numY ::. numX) strikes)

updateParams :: Acc (Vector Double, Vector Double) -> Exp (Double, Double,Double,Double) -> Acc (Matrix Double, Matrix Double, Matrix Double, Matrix Double)
updateParams (T2 myX myY) (T4 tnow alpha beta nu) = T4 myMuX myVarX myMuY myVarY
  where
    I1 numX = shape myX
    I1 numY = shape myY
    -- redundancy fix 2
    myMuY  = generate (Z_ ::. numX ::. numY) (\(I2 x y) -> alpha / toFloating (x*numY+y+1))
    myVarY = generate (Z_ ::. numX ::. numY) (\(I2 x y) -> let r = toFloating (x*numY+y+1) in (nu*nu)/r)
    myMuX  = generate (Z_ ::. numY ::. numX) (\(I2 y x) -> 0.0000001 / toFloating ( (numX+x)*(numY+y) ))
    myVarX = zipWith (\y x -> exp (2*(beta * log x + y - 0.5*nu*nu*tnow)))
              (replicate (Z_ ::. All_ ::. numX) myY)
              (replicate (Z_ ::. numY ::. All_) myX)

-- unused: in Accelerate, you can't `map` this function over matrices. Instead, see tridagParSh
tridagPar :: Acc (Vector Double, Vector Double, Vector Double, Vector Double) -> Acc (Vector Double)
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
tridagParSh :: Shape sh => Acc (Array (sh:.Int) Double, Array (sh:.Int) Double, Array (sh:.Int) Double, Array (sh:.Int) Double) -> Acc (Array (sh:.Int) Double)
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

explicitMethod :: Acc (Vector (Three Double), Vector (Three Double), Matrix Double, Matrix Double, Array DIM3 Double)
               -> Acc (Array DIM3 Double)
explicitMethod (T5 myD myDD myMu myVar result) =
  let I3 s n m = shape result
  in zipWith5
      (\(T3 dx0 dx1 dx2) (T3 dxx0 dxx1 dxx2) mu var (T3 jprev j jnext) ->
        let c1 = (mu*dx0 + 0.5*var*dxx0) * jprev
            c3 = (mu*dx2 + 0.5*var*dxx2) * j
            c2 = (mu*dx1 + 0.5*var*dxx1) * jnext
        in  c1 + c2 + c3)
      (replicate (Z_ ::. s ::. n    ::. All_) myD )
      (replicate (Z_ ::. s ::. n    ::. All_) myDD)
      (replicate (Z_ ::. s ::. All_ ::. All_) myMu)
      (replicate (Z_ ::. s ::. All_ ::. All_) myVar)
      -- Currently using a 3x3x3 stencil and ignoring stuff, 
      -- because Accelerate doesn't export a 1x1x3 stencil. 
      -- Hopefully the other 24 array reads get optimised away.
      (stencil @_ @(Three (Three (Three (Exp Double)))) (\(_,(_,(a,b,c),_),_) -> T3 a b c) (function $ const 0) result)

implicitMethod :: Acc (Vector (Three Double), Vector (Three Double), Matrix Double, Matrix Double, Array DIM3 Double) -> Exp Double -> Acc (Array DIM3 Double)
implicitMethod (T5 myD myDD myMu myVar u) dtInv =
  let Z_ ::. s ::. n ::. m = shape u
      (a, b, c) = unzip3 $ zipWith4 (\mu var (T3 d0 d1 d2) (T3 dd0 dd1 dd2) -> T3
                                    (negate (0.5*(mu*d0 + 0.5*var*dd0)))
                                    (dtInv - 0.5*(mu*d1 + 0.5*var*dd1))
                                    (negate (0.5*(mu*d2 + 0.5*var*dd2))))
                                  (replicate (Z_ ::. s ::. All_ ::. All_) myMu)
                                  (replicate (Z_ ::. s ::. All_ ::. All_) myVar)
                                  (replicate (Z_ ::. s ::. n    ::. All_) myD)
                                  (replicate (Z_ ::. s ::. n    ::. All_) myDD)
  in tridagParSh $ T4 a b c u

rollback :: Exp ( Double, Double)
         -> Acc ( Array DIM3 Double
                , Matrix Double, Vector (Three Double), Vector (Three Double), Matrix Double
                , Matrix Double, Vector (Three Double), Vector (Three Double), Matrix Double)
         -> Acc (Array DIM3 Double)
rollback (T2 tnow tnext) (T9 myResult myMuX myDx myDxx myVarX myMuY myDy myDyy myVarY) =
  let dtInv = 1/(tnext-tnow)

      u1 = explicitMethod (T5 myDx myDxx myMuX myVarX myResult)
      u2 = zipWith (\u_el res_el -> dtInv*res_el+0.5*u_el) u1 myResult

      myResultTr1 = transposeOn _1 _2 myResult
      v = explicitMethod (T5 myDy myDyy myMuY myVarY myResultTr1)
      u3 = zipWith (+) u2 (transposeOn _1 _2 v)

      u4 = implicitMethod (T5 myDx myDxx myMuX myVarX u3) dtInv

      y = zipWith (\u_el v_el -> dtInv*u_el - 0.5*v_el)(transposeOn _1 _2 u4) v
      myResultTr2 = implicitMethod (T5 myDy myDyy myMuY myVarY y) dtInv
  in transposeOn _1 _2 myResultTr2

value :: Exp (Int, Int, Int, Double, Double, Double, Double, Double) -> Acc (Vector Double) -> Acc (Vector Double)
value (T8 numX numY numT s0 t alpha nu beta) strikes =
  let (T2 myXindex myYindex, T3 myX myY myTimeline) = initGrid $ T7 s0 alpha nu t numX numY numT
      (myDx, myDxx) = unzip $ initOperator myX
      (myDy, myDyy) = unzip $ initOperator myY
      myResult1 = setPayoff $ T3 strikes myX myY
      myTimeline_neighbours = reverse $ zip (init myTimeline) (tail myTimeline)
      final = iterateOver
                myTimeline_neighbours
                myResult1
                (\(T2 tnow tnext) res ->
                  let T4 mux varx muy vary = updateParams (T2 myX myY) (T4 tnow alpha beta nu)
                  in rollback (T2 tnow tnext) (T9 res mux myDx myDxx varx muy myDy myDyy vary))
  in slice final (Z_ ::. All_ ::. myYindex ::. myXindex)

main' :: Acc (Scalar Input) -> Acc (Vector Double)
main' s = let (T9 outerloopcount numX numY numT s0 t alpha nu beta) = the s in
  let strikes = generate (I1 outerloopcount) (\(I1 i) -> 0.001 * toFloating i)
  in value (T8 numX numY numT s0 t alpha nu beta) strikes

-- sequential loop over the first argument, updating the second
iterateOver :: (Elt a, Arrays b) => Acc (Vector a) -> Acc b -> (Exp a -> Acc b -> Acc b) -> Acc b
iterateOver xs initial f =
  let Z_ ::. n = shape xs
  in afst $ awhile
    (map (< 1) . asnd)
    (\(T2 current i) -> T2 (f (xs!I1 (the i)) current) (map (+1) i))
    (T2 initial $ unit 0)
