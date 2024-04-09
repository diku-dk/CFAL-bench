{-# LANGUAGE OverloadedStrings #-}

module Quickhull (quickhull, Point) where

import Data.Array.Accelerate hiding (scanr, scanr1, scanl, scanl', scanr')
import Data.Array.Accelerate.Debug.Trace
import qualified Prelude

type Point = (Double, Double)
type Line = (Point, Point)
type Label = Int

type State =
  ( Vector (Point, Label) -- L, the points that should still be handled, combined with the index of their segment
  , Vector Point -- R, the points that are definitely on the convex hull.
  -- Label i corresponds with the points between R[i] and R[i+1]
  )

quickhull :: Acc (Vector Point) -> Acc (Vector Point)
quickhull
  = asnd
  -- While there are undecided points,
  . awhile
      (\(T2 points _) -> unit $ size points /= 0)
      -- Add more points to the hull
      step
  -- After creating an initial state, with the input split in two segments
  . initialize

initialize :: Acc (Vector Point) -> Acc State
initialize points = T2 newPoints hull
  where
    -- Find the leftmost and rightmost points.
    -- If there are multiple leftmost or rightmost points,
    -- we choose the one with the minimum or maximum y coordinate.
    line@(T2 leftMost rightMost)
      = the $ fold1 (\(T2 min1 max1) (T2 min2 max2) -> T2 (min min1 min2) (max max1 max2))
      $ map (\p -> T2 p p) points
    -- Note that this can be done with two folds (one for leftmost and one for rightmost),
    -- but we fused them manually.
  
    -- Partition the array in two segments: the points above the line, and the points below the line.
    -- This is implicit in the description in the paper.
    -- We drop all the points on the line, as they are definitely not on the convex hull.
    -- Preserving these elements may give problems, for instance if all points are on one line.
    -- By doing so, we also drop 'leftMost' and 'rightMost'.

    -- Partitioning happens similar to a filter (compaction).
    -- We use a scan to compute destination indices within a segment,
    -- and then offset the second segment by the size of the first segment.
    offsets :: Acc (Vector (Int, Int))
    counts :: Acc (Scalar (Int, Int))
    T2 offsets counts
      = scanl' (\(T2 l1 l2) (T2 r1 r2) -> T2 (l1 + r1) (l2 + r2)) (T2 0 0)
      $ map f points
      where
        -- Returns (1, 0) if the point is in the upper segment,
        -- (0, 1) if it is in the lower segment,
        -- or (0, 0) otherwise.
        f :: Exp Point -> Exp (Int, Int)
        f p
          = cond (p == leftMost || p == rightMost) (T2 0 0)
          $ cond (d > 0.0) (T2 1 0)
          $ cond (d < 0.0) (T2 0 1)
            (T2 0 0)
          where d = nonNormalizedDistance line p

    T2 countUpper countLower = the counts

    -- Compute the destinations for partitioning
    destination :: Acc (Vector (Maybe DIM1))
    destination = zipWith f points offsets
      where
        f :: Exp Point -> Exp (Int, Int) -> Exp (Maybe DIM1)
        f p (T2 idxUpper idxLower)
          = cond (p == leftMost || p == rightMost) Nothing_
          $ cond (d > 0.0) (Just_ $ I1 idxUpper)
          $ cond (d < 0.0) (Just_ $ I1 $ countUpper + idxLower)
            Nothing_
          where
            d = nonNormalizedDistance line p

    empty :: Acc (Vector (Point, Label))
    empty = fill (I1 $ countUpper + countLower) (T2 (T2 0.0 0.0) 0)

    -- Perform the actual permutation
    newPoints :: Acc (Vector (Point, Label))
    newPoints
      = permute const empty (destination !)
      -- Tuple each point with the index of its segment
      $ map (\p -> T2 p (nonNormalizedDistance line p < 0.0 ? (1, 0))) points

    -- Initial hull consists of only leftMost and rightMost
    hull :: Acc (Vector Point)
    hull = generate (Z_ ::. 2) (\(I1 idx) -> idx == 0 ? (leftMost, rightMost))

step :: Acc State -> Acc State
step (T2 pointsWithLabels hull) = T2 newPoints newHull
  where
    (points, labels) = unzip pointsWithLabels

    -- Use a segmented scan to compute the furthest point to the line of each segment.
    -- Only the value at the last position of a segment is accurate after this scan. 
    furthest :: Acc (Vector Point)
    furthest
      = map snd -- Drop the distance, only keep the point
      $ segmentedScanl1 max labels
      $ map (\(T2 p segment) -> T2 (distance segment p) p) pointsWithLabels

    distance :: Exp Label -> Exp Point -> Exp Double
    distance label point = nonNormalizedDistance (T2 a b) point
      where
        a = hull !! label
        b = hull !! (label == size hull - 1 ? (0, label + 1))

    -- Store furthest point per segment
    furthestPerLabel :: Acc (Vector Point)
    furthestPerLabel = permute const
      -- Use -Infinity as default, to detect empty segments
      -- (parts of the hull where no undecided points remain)
      (generate (shape hull) $ const $ noPoint)
      -- Only write the value if it is the last of a segment
      (\(I1 idx) ->
        (idx == size points - 1 || labels !! idx /= labels !! (idx + 1))
        ? (Just_ $ I1 $ labels !! idx, Nothing_)  
      )
      furthest

    noPoint = T2 (-1.0/0.0) (-1.0/0.0)
  
    -- Mapping from the old indices in the hull to the new indices.
    hullNewIndices :: Acc (Vector Int)
    hullNewSize :: Acc (Scalar Int)
    T2 hullNewIndices hullNewSize = scanl' (+) 0 $ map (\p -> p == noPoint ? (1, 2)) furthestPerLabel

    -- Add furthest points to the hull
    newHull :: Acc (Vector Point)
    newHull = permute
      const
      (permute
        const
        (generate (I1 $ the hullNewSize) $ const $ T2 0.0 0.0)
        (\idx -> Just_ $ I1 $ hullNewIndices ! idx)
        hull
      )
      (\idx -> cond (furthestPerLabel ! idx == noPoint) Nothing_
        $ Just_ $ I1 $ (hullNewIndices ! idx) + 1)
      furthestPerLabel

    -- Filter and reorder the remaining points.
    -- If the corresponding point in the hull of a given segment are A and B,
    -- and the furthest point is F, then we need to first store all the
    -- elements left to the line from A to F and then the points to the left
    -- of the line from F to B. Points within the triangle formed by A, B and F
    -- are dropped.

    -- Characterize whether this point should is left to AF (1), left to
    -- (FB) (2) or neither (0).
    groups :: Acc (Vector Int8)
    groups = map f pointsWithLabels
      where
        f :: Exp (Point, Label) -> Exp Int8
        f (T2 p label)
          = cond (p == f) 0
          $ cond (nonNormalizedDistance (T2 a f) p > 0.0) 1
          $ cond (nonNormalizedDistance (T2 f b) p > 0.0) 2
            0
          where
            a = hull !! label
            b = hull !! (label == size hull - 1 ? (0, label + 1))
            f = furthestPerLabel !! label

    -- Compute the destinations for all points.
    -- First compute the local offsets, within a group, within a segment.
    -- Note that this is an inclusive scan, so the offsets are off by one.
    localOffsets :: Acc (Vector (Int, Int))
    localOffsets
      = segmentedScanl1 (\(T2 a1 a2) (T2 b1 b2) -> T2 (a1 + b1) (a2 + b2)) labels
      $ map (\g -> cond (g == 1) (T2 1 0) $ cond (g == 2) (T2 0 1) (T2 0 0)) groups

    segmentSizes :: Acc (Vector (Int, Int))
    segmentSizes = permute
      const
      (generate (shape hull) $ const (T2 0 0))
      (\(I1 idx) ->
        (idx == size points - 1 || labels !! idx /= labels !! (idx + 1))
        ? (Just_ $ I1 $ labels !! idx, Nothing_)  
      )
      localOffsets
    
    segmentOffsets :: Acc (Vector Int)
    newSize :: Acc (Scalar Int)
    T2 segmentOffsets newSize = scanl' (+) 0 $ map (\(T2 size1 size2) -> size1 + size2) segmentSizes

    destination :: Exp DIM1 -> Exp (Maybe DIM1)
    destination idx
      = cond (g == 1) (Just_ $ I1 $ segmentOffset + leftOffset - 1)
      $ cond (g == 2) (Just_ $ I1 $ segmentOffset + leftSize + rightOffset - 1)
      Nothing_
      where
        g = groups ! idx
        label = labels ! idx
        T2 leftOffset rightOffset = localOffsets ! idx
        segmentOffset = segmentOffsets !! label
        T2 leftSize _ = segmentSizes !! label

    setNewLabel :: Exp (Point, Label) -> Exp Int8 -> Exp (Point, Label)
    setNewLabel (T2 point label) g = T2 point $ (hullNewIndices !! label) + fromIntegral g - 1

    newPoints :: Acc (Vector (Point, Label))
    newPoints
      = permute const (generate (I1 $ the $ newSize) $ const $ T2 (T2 0.0 0.0) 0) destination
      $ zipWith setNewLabel pointsWithLabels groups

-- Labeled inclusive scan.
segmentedScanl1 :: Elt a => (Exp a -> Exp a -> Exp a) -> Acc (Vector Label) -> Acc (Vector a) -> Acc (Vector a)
segmentedScanl1 f labels vector
  -- Drop the flags
  = map snd
  -- Lift 'f' to a segmented operator
  $ scanl1 (segmented f)
  -- Pair the values with flags denoting whether a segment starts there
  $ imap (\(I1 idx) a -> T2 (idx == 0 || labels !! idx /= labels !! (idx - 1)) a) vector

segmentedScanr1 :: Elt a => (Exp a -> Exp a -> Exp a) -> Acc (Vector Label) -> Acc (Vector a) -> Acc (Vector a)
segmentedScanr1 f labels vector
  = map snd
  $ scanr1 (Prelude.flip (segmented (Prelude.flip f)))
  -- Pair the values with flags denoting whether a segment ends there
  $ imap (\(I1 idx) a -> T2 (idx == size vector - 1 || labels !! idx /= labels !! (idx + 1)) a) vector

segmented :: Elt a => (Exp a -> Exp a -> Exp a) -> Exp (Bool, a) -> Exp (Bool, a) -> Exp (Bool, a)
segmented f (T2 aF aV) (T2 bF bV) = T2 (aF || bF) (bF ? (bV, f aV bV))

-- Computes the distance of a point to a line, which is off by a factor depending on the line.
nonNormalizedDistance :: Exp Line -> Exp Point -> Exp Double
nonNormalizedDistance (T2 (T2 x1 y1) (T2 x2 y2)) (T2 x y) = nx * x + ny * y - c
  where
    nx = y1 - y2
    ny = x2 - x1
    c  = nx * x1 + ny * y1

scanr1 f = reverse . scanl1 (flip f) . reverse
scanl' f x xs = T2 (oneOver x $ map (f x) $ scanl1 f xs) (fold f x xs)
oneOver x xs = 
  let (sh ::. sz) = shape xs
      sh' = sh ::. 1
      x' = generate sh' (const x)
      xs' = backpermute (sh ::. (sz-1)) Prelude.id xs
  in x' ++ xs'

