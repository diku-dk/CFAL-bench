import "util"
-- import "mg-in-work"

def relaxNas [n] (a: [4]real) (u_d: [n][n][n]real) : [n][n][n]real =
  let nm1= (i32.i64 n) - 1
  let iterBody (i3: i32) (i2: i32) : [n]real =
      let f (i1: i32) = #[unsafe]
           (  u_d[i3, (i2-1) & nm1, i1] +
              u_d[i3, (i2+1) & nm1, i1] +
              u_d[(i3-1) & nm1, i2, i1] +
              u_d[(i3+1) & nm1, i2, i1]
           ,  
              u_d[(i3-1) & nm1, (i2-1) & nm1, i1] +
              u_d[(i3-1) & nm1, (i2+1) & nm1, i1] +
              u_d[(i3+1) & nm1, (i2-1) & nm1, i1] +
              u_d[(i3+1) & nm1, (i2+1) & nm1, i1]
           )
      let g u1s u2s (i1: i32) = #[unsafe]
           ( a[0] * ( u_d[i3, i2, i1] ) +
             a[1] * ( u_d[i3, i2, (i1-1) & nm1] + u_d[i3, i2, (i1+1) & nm1] + u1s[i1] ) +
             a[2] * ( u2s[i1] + u1s[(i1-1) & nm1] + u1s[(i1+1) & nm1] ) +
             a[3] * ( u2s[(i1-1) & nm1] + u2s[(i1+1) & nm1] )
           )
      let (u1s, u2s) = unzip (tabulate' n f)
      in  tabulate' n (g u1s u2s)
  in  tabulateIntra_2d n n iterBody

-- performance testing
-- ==
-- entry: nasA nasS
-- random input { [512][512][512]f64 [512][512][512]f64 }

entry nasA [n] (v: [n][n][n]real) (u: [n][n][n]f64) : [n][n][n]real =
  relaxNas [-8/3, 0, 1/6, 1/12] u |> map2_3d (-) v

entry nasS [n] (v: [n][n][n]real) (u: [n][n][n]f64) : [n][n][n]real =
  relaxNas [-3/8, 1/32, -1/64, 0] u |> map2_3d (+) v

-- entry origAmV [n] (v: [n][n][n]real) (u: [n][n][n]f64) : [n][n][n]real =
--   map2_3d (-) v (A u)
