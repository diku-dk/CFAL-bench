import "util"
-- import "mg-in-work"

def getElm3d (arr:[][][]real) (i: i32) (j: i32) (k: i32) : real =
  #[unsafe] arr[i, j, k]

def get8thElm3d (arr:[][][]real) (i: i32) (j: i32) (k: i32) : real =
  #[unsafe]
  if (i %% 2) + (j %% 2) + (k %% 2) == 3
  then arr[i//2,j//2,k//2]
  else 0

def relaxNas (n: i64) (getElm : i32 -> i32 -> i32 -> real) (a: [4]real) : [n][n][n]real =
  let nm1= (i32.i64 n) - 1
  let iterBody (i3: i32) (i2: i32) : [n]real =
      let f (i1: i32) = #[unsafe]
           (  getElm i3 ((i2-1) & nm1) i1 +
              getElm i3 ((i2+1) & nm1) i1 +
              getElm ((i3-1) & nm1) i2 i1 +
              getElm ((i3+1) & nm1) i2 i1
           ,  
              getElm ((i3-1) & nm1) ((i2-1) & nm1) i1 +
              getElm ((i3-1) & nm1) ((i2+1) & nm1) i1 +
              getElm ((i3+1) & nm1) ((i2-1) & nm1) i1 +
              getElm ((i3+1) & nm1) ((i2+1) & nm1) i1
           )
      let g u1s u2s (i1: i32) = #[unsafe]
           ( a[0] * ( getElm i3 i2 i1 ) +
             a[1] * ( getElm i3 i2 ((i1-1) & nm1) + getElm i3 i2 ((i1+1) & nm1) + u1s[i1] ) +
             a[2] * ( u2s[i1] + u1s[(i1-1) & nm1] + u1s[(i1+1) & nm1] ) +
             a[3] * ( u2s[(i1-1) & nm1] + u2s[(i1+1) & nm1] )
           )
      let (u1s, u2s) = unzip (tabulate' n f)
      in  tabulate' n (g u1s u2s)
  in  tabulateIntra_2d n n iterBody

-------------------------------------------------------------------------------------------------
-- def relaxNas [n] (a: [4]real) (u_d: [n][n][n]real) : [n][n][n]real =
--   let nm1= (i32.i64 n) - 1
--   let iterBody (i3: i32) (i2: i32) : [n]real =
--       let f (i1: i32) = #[unsafe]
--            (  u_d[i3, (i2-1) & nm1, i1] +
--               u_d[i3, (i2+1) & nm1, i1] +
--               u_d[(i3-1) & nm1, i2, i1] +
--               u_d[(i3+1) & nm1, i2, i1]
--            ,  
--               u_d[(i3-1) & nm1, (i2-1) & nm1, i1] +
--               u_d[(i3-1) & nm1, (i2+1) & nm1, i1] +
--               u_d[(i3+1) & nm1, (i2-1) & nm1, i1] +
--               u_d[(i3+1) & nm1, (i2+1) & nm1, i1]
--            )
--       let g u1s u2s (i1: i32) = #[unsafe]
--            ( a[0] * ( u_d[i3, i2, i1] ) +
--              a[1] * ( u_d[i3, i2, (i1-1) & nm1] + u_d[i3, i2, (i1+1) & nm1] + u1s[i1] ) +
--              a[2] * ( u2s[i1] + u1s[(i1-1) & nm1] + u1s[(i1+1) & nm1] ) +
--              a[3] * ( u2s[(i1-1) & nm1] + u2s[(i1+1) & nm1] )
--            )
--       let (u1s, u2s) = unzip (tabulate' n f)
--       in  tabulate' n (g u1s u2s)
--   in  tabulateIntra_2d n n iterBody
--------------------------------------------------------------------------------------------------

-- performance testing
-- ==
-- entry: nasA nasS
-- random input { [512][512][512]f64 [512][512][512]f64 }

entry nasA [n] (v: [n][n][n]real) (u: [n][n][n]f64) : [n][n][n]real =
  relaxNas n (getElm3d u) [-8/3, 0, 1/6, 1/12] |> map2_3d (-) v

entry nasS [n] (v: [n][n][n]real) (u: [n][n][n]f64) : [n][n][n]real =
  relaxNas n (getElm3d u) [-3/8, 1/32, -1/64, 0] |> map2_3d (+) v

-- performance testing
-- ==
-- entry: nasQ
-- random input { [512][512][512]f64 [256][256][256]f64 }

entry nasQ [n] (v: [2*n][2*n][2*n]real) (u: [n][n][n]f64) : [2*n][2*n][2*n]real =
  relaxNas (2*n) (get8thElm3d u) [1, 1/2, 1/4, 1/8] |> map2_3d (+) v

-- entry origAmV [n] (v: [n][n][n]real) (u: [n][n][n]f64) : [n][n][n]real =
--   map2_3d (-) v (A u)
