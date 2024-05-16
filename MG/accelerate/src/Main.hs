module Main where
import Data.Array.Accelerate
import qualified Data.Array.Accelerate.LLVM.Native as CPU
import qualified Data.Array.Accelerate.LLVM.PTX    as GPU
import Criterion
import Criterion.Main
import qualified Prelude
import Control.Concurrent
import System.IO

main :: Prelude.IO ()
main = do
  setNumCapabilities 1
  -- Prelude.putStrLn $ test @CPU.UniformScheduleFun @CPU.NativeKernel (mg 4 (1,2,3,4))
  Prelude.print $ Prelude.length $ Prelude.show input256
  -- hFlush stderr Prelude.>> hFlush stdout
  Prelude.print $ runN @CPU.Native (mg 4 weightsA) (fromList Z [4]) input256 (fromList (Z :. 256 :.  256 :.  256) $ Prelude.repeat 0)
  -- mg 4 (1,2,3,4) (4, 256, input256)
  -- defaultMain [backend "CPU" $ runN @CPU.Native] --, backend "GPU" GPU.runN]
  defaultMain [backend "CPU" CPU.runN, backend "GPU" GPU.runN]
  where
    makeInput' = CPU.runN makeInput
    input256 = makeInput' $ fromList Z [256]
    input512 = makeInput' $ fromList Z [512]

    backend name runN
      = bgroup name
      $ Prelude.map (testcase runN)
      [(4, 256, input256), (20, 256, input256), (20, 512, input512)]

    testcase runN (iter, n, input) =
      env (Prelude.return $
          runN $ \a ->
              mg n
                (if iter Prelude.== 4 then weightsA else weightsB) (use $ fromList Z [iter])
                a
                (generate (Z_ ::. constant n ::. constant n ::. constant n) $ const 0)
        )
        $ \f -> bench ("n=" Prelude.++ Prelude.show n Prelude.++ ", iter=" Prelude.++ Prelude.show iter) $ nf f input

weightsA, weightsB :: (Double, Double, Double, Double)
weightsA = (-3/8, 1/32, -1/64, 0)
weightsB = (-3/17, 1/33, -1/61, 0)

type Array3 = Array DIM3
type Triple a = (a, a, a)

coarse2fine :: Acc (Array3 Double) -> Acc (Array3 Double)
coarse2fine z = generate (Z_ ::. n*2 ::. n*2 ::. n*2) $ \(I3 i j k) ->
    cond ((i `rem` 2) + (j `rem` 2) + (k `rem` 2) == 3)
      (z ! I3 (i `quot` 2) (j `quot` 2) (k `quot` 2))
      0
  where
    I3 n _ _ = shape z

fine2coarse :: Acc (Array3 Double) -> Acc (Array3 Double)
fine2coarse z = generate (Z_ ::. n `quot` 2 ::. m `quot` 2 ::. k `quot` 2) $ \(I3 i j l) ->
    z ! I3 (i * 2 + 1) (j * 2 + 1) (l * 2 + 1)
  where
    I3 n m k = shape z

relax :: (Double, Double, Double, Double) -> Acc (Array3 Double) -> Acc (Array3 Double)
relax = \(w1, w2, w3, w4) ->
  stencil
    (\(x, y, z) ->
      relax2D (w2, w3, w4) x + relax2D (w1, w2, w3) y + relax2D (w2, w3, w4) z)
    mirror
  where
    relax2D :: (Double, Double, Double) -> Triple (Triple (Exp Double)) -> Exp Double
    relax2D (w1, w2, w3) (x, y, z)
      = relax1D (w2, w3) x + relax1D (w1, w2) y + relax1D (w2, w3) z

    relax1D :: (Double, Double) -> Triple (Exp Double) -> Exp Double
    relax1D (w1, w2) (x, y, z) = constant w2 * x + constant w1 * y + constant w2 * z

p :: Acc (Array3 Double) -> Acc (Array3 Double)
p = fine2coarse . relax (1/2, 1/4, 1/8, 1/16)

q :: Acc (Array3 Double) -> Acc (Array3 Double)
q = relax (1, 1/2, 1/4, 1/8) . coarse2fine

a :: Acc (Array3 Double) -> Acc (Array3 Double)
a = relax (-8/3, 0, 1/6, 1/12)

-- The size of all dimensions of the input should be 'n'.
-- 'n' must be a power of two.
m :: Int -> (Double, Double, Double, Double) -> Acc (Array3 Double) -> Acc (Array3 Double)
m 2 weights r = relax weights r -- Base case
m n _ _
  | n Prelude.<= 3 = error "Illegal size"
m n weights r = z'
  where
    rs = p r
    zs = m (n `Prelude.div` 2) weights rs
    z = q zs
    r' = zipWith (-) r $ a z
    z' = zipWith (+) z $ relax weights r

l2 :: Acc (Array3 Double) -> Exp Double
l2 xsss =
  sqrt (
    the (sum $ map (**2) $ reshape (Z_ ::. sz) xsss)
    / fromIntegral sz
  )
  where
    sz = size xsss

mg :: Int -> (Double, Double, Double, Double) -> Acc (Scalar Int) -> Acc (Array3 Double) -> Acc (Array3 Double) -> Acc (Scalar Double)
mg n weights iter v = unit . l2 . zipWith (-) v . a . asnd . awhile
  (\(T2 i _) -> unit $ the i < the iter)
  (\(T2 i u) ->
    let
      r = zipWith (-) v $ a u
      r' = m n weights r
      i' = unit $ the i + 1
    in T2 i' $ zipWith (+) u r'
  )
  . T2 (unit 0)

awhile' ::  Arrays a => (Acc a -> Acc (Scalar Bool)) -> (Acc a -> Acc a) -> Acc a -> Acc a
awhile' cond step init = step $ step $ step $ step $ step $ step init

makeInput :: Acc (Scalar Int) -> Acc (Array3 Double)
makeInput n = generate (Z_ ::. the n ::. the n ::. the n) $ \idx ->
  cond (Prelude.foldl1 (||) $ Prelude.map (== idx) negatives)
    (-1)
    $ cond (Prelude.foldl1 (||) $ Prelude.map (==idx) positives)
      1
      0
  where
    negatives = [
      I3 211 154 98,
      I3 102 138 112,
      I3 101 156 59,
      I3 17 205 32,
      I3 92 63 205,
      I3 199 7 203,
      I3 250 170 157,
      I3 82 184 255,
      I3 154 162 36,
      I3 223 42 240]
    positives = [
      I3 57 120 167,
      I3 5 118 175,
      I3 176 246 164,
      I3 45 194 234,
      I3 212 7 248,
      I3 115 123 207,
      I3 202 83 209,
      I3 203 18 198,
      I3 243 172 14,
      I3 54 209 40]
