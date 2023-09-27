type real = f64

def tabulate' n f = tabulate n (\i -> f (i32.i64 i))

def imapIntra as f =
    #[incremental_flattening(only_intra)] map f as

def tabulate_nest0 n f =
  tabulate_3d n 2 n (\i i0 j -> imapIntra (iota 2) (\j0 -> f (i32.i64 i) (i32.i64 i0) (i32.i64 j) (i32.i64 j0) ))

def tabulate_nest n f =
  imapIntra (iota ((2*n)*(2*n)))
    (\ ij -> let (n, ij) = (i32.i64 n, i32.i64 ij)
             let (ii, jj)  = ( ij / (2*n), ij & (2*n-1) )
             let (i, i0, j, j0) = (ii >> 1, ii & 1, jj >> 1, jj & 1)
             in  f i i0 j j0
    )
             
def Qnas1 [n] (z_d: [n][n][n]real) (u_d: [2*n][2*n][2*n]f64) : [2*n][2*n][2*n]real =
  let nm1= (i32.i64 n) - 1
  let iterBody (i3: i32) (i0: i32) (i2: i32) (j0: i32) : [2*n]real =
      let f1 (i1: i32) = -- #[unsafe] 
        z_d[i3, i2, i1]
      let f2 (i1: i32) = -- #[unsafe] 
        ( z_d[i3, (i2+1) & nm1, i1] + z_d[i3, i2, i1] )
      let f3 (i1: i32) = -- #[unsafe]
        ( z_d[(i3+1) & nm1, i2, i1] + z_d[i3, i2, i1] )
      let f4 (i1: i32) = -- #[unsafe]
        let tmp1 = z_d[i3, i2, i1]
        let tmp2 = z_d[(i3+1) & nm1, i2, i1]
        let r1   = z_d[i3, (i2+1) & nm1, i1] + tmp1
        -- let r2   = tmp2 + tmp1
        let r3   = tmp2 + r1 + z_d[(i3+1) & nm1, (i2+1) & nm1, i1]
        in  r3
      
      let g1 z321s (i1: i32) = -- #[unsafe]
          -- double z321=z_device[i3*mm2*mm1+i2*mm1+i1];
		  -- u_device[2*i3*n2*n1+2*i2*n1+2*i1]+=z321;
		  -- u_device[2*i3*n2*n1+2*i2*n1+2*i1+1]+=0.5*(z_device[i3*mm2*mm1+i2*mm1+i1+1]+z321);
          let z321 = 1.0 * z321s[i1]
          let v2 = 0.5 * (z321 + z321s[(i1+1) & nm1]) -- z_d[i3, i2, (i1+1) & nm1])
          in  [ z321 + u_d[2*i3,2*i2,2*i1], v2 + u_d[2*i3,2*i2,2*i1+1] ] 
      let g2 z1s (i1: i32) = -- #[unsafe]
          -- u_device[2*i3*n2*n1+(2*i2+1)*n1+2*i1]+=0.5*z1[i1];
		  -- u_device[2*i3*n2*n1+(2*i2+1)*n1+2*i1+1]+=0.25*(z1[i1]+z1[i1+1]);
          let (z1, z1p1) = ( z1s[i1], z1s[(i1+1) & nm1] )
          in  [0.5 * z1 + u_d[2*i3,2*i2+1,2*i1], 0.25 * (z1 + z1p1) + u_d[2*i3,2*i2+1,2*i1+1]] 
      let g3 z2s (i1: i32) = -- #[unsafe]
          -- u_device[(2*i3+1)*n2*n1+2*i2*n1+2*i1]+=0.5*z2[i1];
		  -- u_device[(2*i3+1)*n2*n1+2*i2*n1+2*i1+1]+=0.25*(z2[i1]+z2[i1+1]);
          let (z2, z2p1) = ( z2s[i1], z2s[(i1+1) & nm1] )
          in  [0.5 * z2 + u_d[2*i3+1,2*i2,2*i1], 0.25 * (z2 + z2p1) + u_d[2*i3+1,2*i2,2*i1+1]]
      let g4 z3s (i1: i32) = -- #[unsafe]
          -- u_device[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1]+=0.25*z3[i1];
		  -- u_device[(2*i3+1)*n2*n1+(2*i2+1)*n1+2*i1+1]+=0.125*(z3[i1]+z3[i1+1]);
          let (z3, z3p1) = ( z3s[i1], z3s[(i1+1) & nm1] )
          in  [0.25 * z3 + u_d[2*i3+1,2*i2+1,2*i1], 0.125 * (z3 + z3p1) + u_d[2*i3+1,2*i2+1,2*i1+1]] 

      in match (i0, j0) 
            case (0,0) -> tabulate' n (g1 (tabulate' n f1) ) |> flatten |> sized (2*n)
            case (0,1) -> tabulate' n (g2 (tabulate' n f2) ) |> flatten |> sized (2*n)
            case (1,0) -> tabulate' n (g3 (tabulate' n f3) ) |> flatten |> sized (2*n)
            case _     -> tabulate' n (g4 (tabulate' n f4) ) |> flatten |> sized (2*n)
  in  tabulate_nest n iterBody |> unflatten

def Qnas [n] (z_d: [n][n][n]real) =
  let zeros = replicate (2*n) 0 |> replicate (2*n) |> replicate (2*n)
  in  Qnas1 z_d zeros
  
-- performance testing
-- ==
-- entry: nasInterp
-- random input { [256][256][256]f64 [512][512][512]f64 }

entry nasInterp [n] (z: [n][n][n]real) (p: [2*n][2*n][2*n]f64) : [2*n][2*n][2*n]real =
  Qnas1 z p

