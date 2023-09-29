-- import "mg-in-work"

type real = f64

def map2_3d f = map2 (map2 (map2 f))

def tabulate' n f = tabulate n (\i -> f (i32.i64 i))

def imapIntra as f =
    #[incremental_flattening(only_intra)] map f as

def tabulateIntra_2d n2 n1 f =
  tabulate' n2 (\i2 -> imapIntra (iota n1) (\i1 -> f i2 (i32.i64 i1)))

def mAnas [n] (a: [4]real) (v_d: [n][n][n]real) (u_d: [n][n][n]real) : [n][n][n]real =
  let nm1= (i32.i64 n) - 1
  let iterBody (i3: i32) (i2: i32) : [n]real =
      let f (i1: i32) = #[unsafe]
           let u1 = u_d[i3, (i2-1) & nm1, i1] +
                    u_d[i3, (i2+1) & nm1, i1] +
                    u_d[(i3-1) & nm1, i2, i1] +
                    u_d[(i3+1) & nm1, i2, i1]
           let u2 = u_d[(i3-1) & nm1, (i2-1) & nm1, i1] +
                    u_d[(i3-1) & nm1, (i2+1) & nm1, i1] +
                    u_d[(i3+1) & nm1, (i2-1) & nm1, i1] +
                    u_d[(i3+1) & nm1, (i2+1) & nm1, i1]
           in  (u1, u2)
      let g u1s u2s (i1: i32) = #[unsafe]
           -- v_d[i3,i2,i1] -
           ( a[0] * u_d[i3, i2, i1] +
             a[2] * ( u2s[i1] + u1s[(i1-1) & nm1] + u1s[(i1+1) & nm1] ) +
             a[3] * ( u2s[(i1-1) & nm1] + u2s[(i1+1) & nm1] )
           )
      let (u1s, u2s) = map f (map i32.i64 (iota n)) |> unzip
      in  map (g u1s u2s) (map i32.i64 (iota n))
  in  tabulateIntra_2d n n iterBody |> map2_3d (-) v_d

  
-- performance testing
-- ==
-- entry: nasAmV origAmV
-- random input { [512][512][512]f64 [512][512][512]f64 }

entry nasAmV [n] (v: [n][n][n]real) (u: [n][n][n]f64) : [n][n][n]real =
  mAnas [-8/3, 0, 1/6, 1/12] v u

-- entry origAmV [n] (v: [n][n][n]real) (u: [n][n][n]f64) : [n][n][n]real =
--   map2_3d (-) v (A u)
