module Main where

-- import Debug.Trace

main :: IO ()
main = do
  -- perfect split
  let tree = buildTree 2 [[4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]
  print tree

  -- one duplicate [4.0, 4.0]
  let tree = buildTree 3 [[4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0], [8.0, 8.0], [6.0, 6.0], [7.0, 7.0], [4.0, 4.0]]
  print tree

  -- awkward one:
  let tree = buildTree 3 [[4.0, 4.0], [3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]
  print tree

infty :: Float
infty = 1.0 / 0.0

-- | Finds the `k`-th smallest element in `as`
quickSelect :: [Float] -> Int -> Float
quickSelect [] _ = infty
quickSelect as k =
  let p = head as
      -- \^ the pivot should be randomly selected
      as_lth_p = filter (< p) as
      as_eqt_p = filter (== p) as
      as_gth_p = filter (> p) as
   in if k < length as_lth_p
        then quickSelect as_lth_p k
        else
          if k < length as_lth_p + length as_eqt_p
            then p
            else
              quickSelect as_gth_p $
                k - (length as_lth_p + length as_eqt_p)

-- | Constructs the kd-tree in a breadth-first fashion,
--   in a manner similar to running a do loop:
--     for i = 0 .. h-1 do
--       map doIter ...
mkLevel ::
  Int ->
  Int ->
  Int ->
  [(Int, Float)] ->
  [[([Float], Int)]] ->
  ([(Int, Float)], [[([Float], Int)]])
mkLevel _ h i meds refs
  | h == i = (meds, refs)
mkLevel d h i meds refs =
  let (new_meds, refs') = unzip (map doIter refs)
   in mkLevel d h (i + 1) (meds ++ new_meds) (concat refs')
  where
    minmax :: (Float, Float) -> (Float, Float) -> (Float, Float)
    minmax (a1, a2) (b1, b2) = (min a1 b1, max a2 b2)

    maxind :: (Float, Int) -> (Float, Int) -> (Float, Int)
    maxind (a, i) (b, j) =
      case (a < b, a == b) of
        (True, _) -> (b, j)
        (False, False) -> (a, i)
        _ -> (a, min i j)

    -- \| partition a list of points(-index pairs) into a list
    --     containing two sublists: the ones smaller than the
    --     median element of the dimension of widest spread,
    --     and the ones greater than that median element.
    doIter :: [([Float], Int)] -> ((Int, Float), [[([Float], Int)]])
    doIter [] = ((0, infty), [[], []])
    doIter node_pts =
      -- 1. compute (the index of) the dimension of widest spread.
      --    In all uses foldl is equivalent with reduce; its operators
      --    are associative and commutative.
      let (pts, inds) = unzip node_pts
          ne = zip (replicate d infty) (replicate d (-infty))
          (mins, maxs) = unzip $ foldl (zipWith minmax) ne $ zipWith zip pts pts
          dists = map abs $ zipWith (-) maxs mins
          (_, med_dim) = foldl maxind (-infty, d + 1) $ zip dists [0 .. d - 1]

          -- 2. find the median element:
          med_pt =
            quickSelect (map (!! med_dim) pts) $
              length pts `div` 2
          -- 3. partition the points into the ones that are less than
          --    and greater than the median element:
          lthMed (vct, _) = (vct !! med_dim) < med_pt
          geqMed (vct, _) = (vct !! med_dim) >= med_pt
          node_pts' = [filter lthMed node_pts, filter geqMed node_pts]
       in ((med_dim, med_pt), node_pts')

-- | Input:
--     `height` denotes the height of the k-d tree
--     `refs` is an array of length `n` of `d`-dimensional
--            reference points, from which the kd-tree is
--            constructed.
--   Result:
--     `med_dim_pts` is an array of length `2^{height} - 1` of tuples
--                   recording, for each internal node in the kd tree,
--                   the index of the split dimension and the median
--                   value on that dimension.
--     `leaves` is an (iregular) array of (sub)arrays, containing
--              the partition of reference points according to the
--              kd-tree. `leaves` always consists of `2^{height}`
--              subarrays --- because a kd-tree is perfectly
--              balanced --- but the subarrays may be imbalanced,
--              i.e., contain a (slightly) different number of points.
--              Each point is tupled with an integral denoting the
--              (original) position of that point in `refs`.
buildTree ::
  Int ->
  [[Float]] ->
  ([(Int, Float)], [[([Float], Int)]])
buildTree _ [] = ([], [])
buildTree height refs =
  let d = length (head refs)
      (med_dim_pts, leaves) =
        mkLevel d height 0 [] [zip refs [0 .. length refs - 1]]
   in (med_dim_pts, leaves)
