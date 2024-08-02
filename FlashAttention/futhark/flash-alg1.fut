def imap5 as bs cs ds es f = map5 f as bs cs ds (map i32.i64 es)
def imap as f = map f (map i32.i64 as)
def imap2 as bs f = map2 f as (map i32.i64 bs)
def imap2seq ass is f = #[sequential] map2 f ass (#[sequential]map i32.i64 is)

def copy2shr [m][n] (xss: [m][n]f32) : *[m][n]f32 =
  let xss' = copy xss
  in  if opaque(true)
      then xss'
      else let xss'[0,0] = 0f32
           in xss'

def softmax [T][R]  (P: [T*T][R][R]f32) (mis: [T*R]f32) (lis: [T*R]f32) 
                  : ([T*R]f32, [T*R]f32, [T*R]f32, [T*T][R][R]f32) =
  -- segmented reduce with max
  let fmaxseq (ass: [R][R]f32) =
        #[sequential] map (#[sequential]reduce f32.max f32.lowest) ass
  let maxs0 = map fmaxseq P |> unflatten -- : [T][T][R]f32
  let maxs' = loop maxs = replicate T (replicate R f32.lowest)
                for i < R do
                  let maxs[:,i] = map (reduce f32.max f32.lowest) maxs0[:,i]
                  in  maxs
  let maxs = flatten maxs'
  -- segmented reduce with add and updating the Ps
  let faddseq (Pmat : [R][R]f32) (tid: i32) : ([R]f32, [R][R]f32) =
      imap2seq Pmat (iota R) (\ Prow i ->
        let tidy = tid / i32.i64 T 
        let row_max = maxs[tidy*(i32.i64 R) + i]
        let Pijs = 
             imap2seq Prow (iota R)
              (\ pij _j -> f32.exp (pij - row_max))
        let sumsi = #[sequential] reduce (+) 0.0f32 Pijs
        in  (sumsi, Pijs)
      ) |> unzip
   let (sums0, P) = map2 faddseq P (map i32.i64 (iota (T*T))) -- : [T][T][R]f32
                 |> unzip
   let sums0 = unflatten sums0
   let sums' = loop sums = replicate T (replicate R 0.0f32)
                for i < R do
                  let sums[:,i] = map (reduce (+) 0) sums0[:,i]
                  in  sums
   let sums = flatten sums'
   -- the rest
   let (mis,lis,ess,els) = unzip4 <|
     imap5 mis lis maxs sums (iota (T*R))
       (\ mi_old li_old maxi sumi _tid ->
         let mi_new = if mi_old > maxi then mi_old else maxi
         let eij = f32.exp (maxi - mi_new)
         let eli = li_old * f32.exp (mi_old - mi_new)
         let li_new = eli + sumi * eij
         in  (mi_new, li_new, eij, eli)
       )
   let P = imap2 P (iota (T*T))
     (\Pmat tid ->
         imap2seq Pmat (iota R)
           (\Prow i ->
             imap2seq Prow (iota R)
               (\ pij _j -> ess[tid / (i32.i64 T) * (i32.i64 R) + i] * pij )
           )
     )
   in (mis,lis,els,P)

def FlashAttention [S][T][R][E]  -- N = S * T * R & d = E * T * R
        (K: [S*T*R][E*T*R]f32) 
        (V: [S*T*R][E*T*R]f32)
        (Q: [T*R][E*T*R]f32) 
      : [E][T*T][R][R]f32 =
  let B = T * R
  let (TT, RR, EE, BB) = (i32.i64 T, i32.i64 R, i32.i64 E, i32.i64 B) 
  let mis = replicate (T*R) f32.lowest
  let lis = replicate (T*R) 0.0f32
  
  -- copy Q[bid:bid+1] from global to shared
  let Qsh = copy2shr Q
  let O = replicate R 0.0f32 -- : [T*T][E][R][R]f32
       |> replicate R
       |> replicate (T*T)
       |> replicate E
       
  let (O, _, _) = 
    loop (O, mis, lis) for block_col < S do
       let P = replicate R 0.0f32
            |> replicate R
            |> replicate (T*T)
       let P =
         loop P for k < EE do
           let kk = i64.i32 k
           let Ksh = replicate (T*R+1) 0.0f32
                   |> replicate (T*R)
           let K' = K[block_col*B : block_col*B + T*R][kk*B : kk*B + T*R]
           let Ksh[:,0:T*R] = K'
           -- let Ksh = K'
           let ff (Preg: [R][R]f32) (tidd: i64) =
                let tid = i32.i64 tidd
                let tidy = tid / TT
                let tidx = tid - tidy * TT
                in
                loop Preg = copy Preg for idx < BB do
                  loop Preg for i < RR do
                    loop Preg for j < RR do
                      let Preg[i,j] = Preg[i,j] +
                                      Ksh[tidx * RR + j][idx] *
                                      Qsh[tidy * RR + i][k*BB + idx]
                      in  Preg
                                      
           in  map2 ff P (iota (T*T))
       -- softmax
       let (mis,lis,els,P) = softmax P mis lis
       -- publish P in [T*R][T*R] form
       let Psh = 
         loop Psh = replicate ((T*R)*(T*R)) 0.0f32
         for i < RR do
           loop Psh for j < RR do
             let (is, Ps) = 
                imap (iota (T*T)) 
                  (\tid -> let tidy = tid / TT
                           let tidx = tid - tidy*TT
                           in (i64.i32 ((tidy * RR + i) * BB + tidx * RR + j), P[tid][i][j])
                  ) |> unzip
             in scatter Psh is Ps
       let O =
         loop O for k < EE do
           let kk = i64.i32 k
           let Vsh = replicate (T*R+1) 0.0f32
                  |> replicate (T*R)
           let V' = V[block_col*B : block_col*B + T*R][kk*T*R : kk*T*R + T*R]
           let Vsh[:,0:T*R] = V'
           -- let Vsh = V'
           -- do update to O
           let fOupd (Omat: [R][R]f32) (tidd: i64) =
             let tid = i32.i64 tidd
             let tidy = tid / TT
             let tidx = tid - tidy * TT
             let row_offset = tidy * RR
             in
               imap2seq Omat (iota R)
                 (\Orow i ->
                   imap2seq Orow (iota R)
                     (\Oij j ->
                        let ii = row_offset + i
                        let Oij = Oij * els[ii]
                        let Oij =
                          loop Oij for idx < BB do
                             Oij + Psh[ii * BB + idx] *  
                                   Vsh[idx, tidx * RR + j]
                        in Oij / lis[ii]
                     )
                 )
           let O[k] = map2 fOupd O[k,:,:,:] (iota (T*T))
           in  O
   
       in  (O, mis, lis)

  in O

--  let O' = transpose O          -- : [E][T*T ][R][R]f32
--        |> unflatten O          -- : [T][T][E][R][R]f32
--        |> map transpose O      -- : [T][E][T][R][R]f32
--        |> map (map transpose)  -- : [T][E][R][T][R]f32
--        |> map transpose        -- : [T][R][E][T][R]f32
--        |> flatten              -- : [T*R ][E][T][R]f32
--        |> map flatten          -- : [T*R ][E*T ][R]f32 
--        |> map flatten          -- : [T*R][E*T*R]f32
--  in O'


entry mk_input (s:i64) (e:i64) (b:i64) : ([s][b][e*b]f32, [s*b][e*b]f32, [s*b][e*b]f32) =
  -- (n, d) = (s*b, e*b)
  let Q = replicate (e*b) 1.0 |> replicate b |> replicate s
  let K = replicate (e*b) 1.0 |> replicate (s*b)
  let V = replicate (e*b) 1.0 |> replicate (s*b)
  in  (Q, K, V)

--
-- ==
-- entry: main16
-- "Class 256-16"  script input { (mk_input 256i64 1i64 16i64) }
-- "Class 512-16"  script input { (mk_input 512i64 1i64 16i64) }
entry main16 [S] (Q: [S][8*2][1*8*2]f32) (K: [S*8*2][1*8*2]f32) (V: [S*8*2][1*8*2]f32) =
  #[incremental_flattening(only_intra)]
  map (FlashAttention K V) Q


--
-- ==
-- entry: main64
-- "Class 256-64"  script input { (mk_input 256i64 1i64 64i64) }
-- "Class 512-64"  script input { (mk_input 512i64 1i64 64i64) }
entry main64 [S] (Q: [S][16*4][1*16*4]f32) (K: [S*16*4][1*16*4]f32) (V: [S*16*4][1*16*4]f32) =
  #[incremental_flattening(only_intra)]
  map (FlashAttention K V) Q

--
-- ==
-- entry: main128
-- "Class 64-128"  script input { (mk_input 128i64 2i64 64i64) }
-- "Class 128-128" script input { (mk_input 256i64 2i64 64i64) }
entry main128 [S] (Q: [S][16*4][2*16*4]f32) (K: [S*16*4][2*16*4]f32) (V: [S*16*4][2*16*4]f32) =
  #[incremental_flattening(only_intra)]
  map (FlashAttention K V) Q


