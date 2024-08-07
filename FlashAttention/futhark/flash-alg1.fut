import "intrinsics-accs"

def imap5 as bs cs ds es f = map5 f as bs cs ds (map i32.i64 es)
def imap as f = map f as
def imap2 as bs f = map2 f as (map i32.i64 bs)
def imap2seq ass is f = #[sequential] map2 f ass (#[sequential]map i32.i64 is)
def imap2seq' ass is f = #[sequential] map2 f ass is
def imapseq is f = #[sequential] map f (#[sequential]map i32.i64 is)

def copy2shr [m][n] (xss: [m][n]f32) : *[m][n]f32 = #[unsafe]
  let xss' = copy xss
  in  if opaque(true) then xss'
      else xss' with [0,0] = 0f32

def accUpdateO [Q] (T: i64) (R: i64) (k: i32) (Ok: [T*T][R*R+1]f32) (O: *[Q]f32) : *[Q]f32 =
  let (TT, RR) = (i32.i64 T, i32.i64 R)
  let inner = RR*RR+1
  let glb_offset = k * (TT*TT*inner)
  let f (Oacc: *acc ([Q]f32)) tid =
    let tid = i32.i64 tid
    let offset = glb_offset + tid*inner in
    loop Oacc for i < RR do
      loop Oacc for j < RR do
        let elm = Ok[tid][i*RR+j]
        let ind = (offset + i*RR) + j
        in  write Oacc (i64.i32 ind) elm
  in scatter_stream O f (iota (T*T))


def copy2shrPad (T: i64) (R: i64) 
                (Ksh: *[(T*R)*(T*R+1)]f32)
                (Kgb: [T*R][T*R]f32)
              : *[(T*R)*(T*R+1)]f32 = #[unsafe]
  let (TT, RR, BB) = (i32.i64 T, i32.i64 R, i32.i64 (T*R))
  let f (Kacc: *acc ([(T*R)*(T*R+1)]f32)) (tid: i64) : acc ([(T*R)*(T*R+1)]f32) =
    let tid = i32.i64 tid in
    loop Kacc for i < RR*RR do
      let line_in_chunk = tid / BB
      let elem_in_line  = tid % BB
      let line_offset = i * (TT/RR)
      let elm = #[unsafe] Kgb[ line_offset + line_in_chunk, elem_in_line]
      let ind = (line_offset + line_in_chunk)*(BB+1) + elem_in_line
      in  write Kacc (i64.i32 ind) elm
  in scatter_stream Ksh f (iota (T*T))    
  
def segred [T][R] (bop: f32 -> f32 -> f32) (ne: f32) (P: [T*T][R*R+1]f32) (tmp: *[(T*R)*(T*R+1)]f32) (res: *[T*R+1]f32) = #[unsafe]
  let (TT, RR) = (i32.i64 T, i32.i64 R)
  -- first reduction
  let f (Kacc: *acc ([(T*R)*(T*R+1)]f32)) (tid: i64) : acc ([(T*R)*(T*R+1)]f32) =
    let tid = i32.i64 tid in
    loop Kacc for i < RR do
      let r = loop r = ne for j < RR do
                bop r P[tid][i*RR+j]
      let ind = i64.i32 (tid * (RR+1) + i)
      in  write Kacc ind r
  let tmp = scatter_stream tmp f (iota (T*T))
  
  -- second reduction
  let vs =
    map(\ tid ->
         let tid = i32.i64 tid
         let tidy = tid / RR
         let tidr = tid % RR
         in
         loop r = ne for i < TT do
           bop r tmp[tidy*(TT*(RR+1)) + i*(RR+1) + tidr] 
       ) (iota (T*R))
  -- let res[:T*R] = vs
  let res = scatter res (iota (T*R)) vs
  in (tmp, res)

def mapScat [B] (as: [B]f32) (bs: [B]f32) (cs: [B]f32) (res: *[B]f32) (f: f32 -> f32 -> f32 -> f32) : [B]f32 =
  scatter res (iota B) (map3 f as bs cs)

def softmax'' [T][R] (KV: *[(T*R)*(T*R+1)]f32) (P: *[T*T][R*R+1]f32)
                    (maxs: *[T*R+1]f32) (sums: *[T*R+1]f32)
                    (ess: *[T*R+1]f32) (els: *[T*R+1]f32)
                    (mis: *[T*R+1]f32) (lis: *[T*R+1]f32)
                  : *([(T*R)*(T*R+1)]f32, [T*R+1]f32, [T*R+1]f32, [T*R+1]f32, [T*R+1]f32, [T*R+1]f32, [T*R+1]f32, [T*T][R*R+1]f32) = 
  (KV, mis, lis, els, ess, maxs, sums, P)


def softmax' [T][R] (KV: *[(T*R)*(T*R+1)]f32) (P: *[T*T][R*R+1]f32)
                    (maxs: *[T*R+1]f32) (sums: *[T*R+1]f32)
                    (ess: *[T*R+1]f32) (els: *[T*R+1]f32)
                    (mis: *[T*R+1]f32) (lis: *[T*R+1]f32)
                  : *([(T*R)*(T*R+1)]f32, [T*R+1]f32, [T*R+1]f32, [T*R+1]f32, [T*R+1]f32, [T*R+1]f32, [T*R+1]f32, [T*T][R*R+1]f32) = #[unsafe]
  --
  let RR = i32.i64 R
  let (KV, maxs) = segred (f32.max) (f32.lowest) P KV maxs
  let maxs = opaque maxs
  --
  let f (Porg: [R*R+1]f32) (tid: i32) : *[R*R+1]f32 = #[unsafe]
    let Pmat = imapseq (iota (R*R+1)) 
                 (\i -> if i == RR*RR then 0.0f32 else Porg[i] )
    let tidy = tid / i32.i64 T
    in
    loop Pmat for i < R do
      let row_max = #[unsafe] maxs[tidy*(i32.i64 R) + i32.i64 i] in
      loop Pmat for j < R do
        let Pmat[i*R+j] = f32.exp (Pmat[i*R+j] - row_max)
        in  Pmat
    --
  let P = map i32.i64 (iota (T*T)) |> map2 f P
  --
  let (KV, sums) = segred (+) 0f32 P KV sums
  --
  let (mis', lis', ess', els') = unzip4 <|
     imap (iota (T*R))
       (\ tid -> #[unsafe]
         let (mi_old, li_old, maxi, sumi) = 
            (mis[tid], lis[tid], maxs[tid], sums[tid]) 
         let mi_new = if mi_old > maxi then mi_old else maxi
         let eij = f32.exp (maxi - mi_new)
         let eli = li_old * (f32.exp (mi_old - mi_new))
         let li_new = eli + sumi * eij
         in  (mi_new, li_new, eij, eli)
       )
  let mis[:T*R] = mis'
  let lis[:T*R] = lis'
  let ess[:T*R] = ess'
  let els[:T*R] = els'
  --
  let P = imap2 P (iota (T*T))
     (\Porg tid ->
         let Pmat = imapseq (iota (R*R+1)) 
                      (\i -> if i == RR*RR then 0.0f32 else Porg[i] )
         let ess_offs = (tid / (i32.i64 T)) * (i32.i64 R) in
         loop Pmat for i < R do
           let ind_ess = ess_offs + i32.i64 i
           let ess_elm = #[unsafe] ess[ind_ess] in
           loop Pmat for j < R do
             let Pmat[i*R+j] = ess_elm * Pmat[i*R+j]
             in  Pmat
     )
  --
  in (KV, mis, lis, els, ess, maxs, sums, P)


def FlashAttention [S][T][R][E]  -- N = S * T * R & d = E * T * R
        (K: [S*T*R][E*T*R]f32) 
        (V: [S*T*R][E*T*R]f32)
        (Q: [T*R][E*T*R]f32) = #[unsafe]
      -- : [E][T*T][R][R]f32 = #[unsafe]
  let B = T * R
  let (TT, RR, EE, BB) = (i32.i64 T, i32.i64 R, i32.i64 E, i32.i64 B) 
  let mis  = replicate (T*R+1) f32.lowest
  let lis  = replicate (T*R+1) 0.0f32
  let maxs = replicate (T*R+1) f32.lowest
  let sums = replicate (T*R+1) 0.0f32
  let ess  = replicate (T*R+1) 0.0f32
  let els  = replicate (T*R+1) 0.0f32
  
  -- copy Q[bid:bid+1] from global to shared
  let Qsh = copy2shr Q

  let O = replicate (E * (T*T) * (R*R+1)) 0.0f32 -- result of type [E][T*T][R][R]f32

  let KVsh = replicate ((T*R)*(T*R+1)) 0.0f32

  let (O : *[E * (T*T) * (R*R+1)]f32, _mis, _lis, _maxs, _sums, _, _, _) = 
    loop (O, mis, lis, maxs, sums, ess, els, KVsh)
    for block_col < S do
       let P = replicate (R*R+1) 0.0f32
            |> replicate (T*T)
       let (P, KVsh) =
         #[unroll]
         loop (P, KVsh) for k < EE do
           let kk = i64.i32 k
           let K' = K[block_col*B : block_col*B + T*R, kk*B : kk*B + T*R] :> [T*R][T*R]f32
           -- let KVsh[:, 0:T*R] = K'
           let KVsh = copy2shrPad T R KVsh K'
           let ff Porg (tidd: i64) = #[unsafe]
                let tid = i32.i64 tidd
                let tidy = tid / TT
                let tidx = tid % TT
                -- let Preg = copy Porg
                let Preg = imapseq (iota (R*R+1)) 
                             (\i -> if i == RR*RR then 0.0f32 else Porg[i] )
                in
                loop Preg for idx < BB do
                  loop Preg for i < R do
                    loop Preg for j < R do
                      let Preg[i*R+j] = Preg[i*R+j] +
                      -- let Preg[i,j] = Preg[i,j] +
                                      #[unsafe] KVsh[ (tidx * RR + i32.i64 j)*(BB+1) + idx] *
                                      #[unsafe] Qsh[tidy * RR + i32.i64 i][k*BB + idx]
                      in  Preg
           --
           let P = map2 ff P (iota (T*T))
           in ( P, KVsh )

       ----------------
       ---  SOFTMAX ---
       ----------------
       let (KVsh, mis, lis, els, ess, maxs, sums, P) = 
               softmax' KVsh P maxs sums ess els mis lis
       ----------------
       let (O, KVsh) =
         #[unroll]
         loop (O, KVsh) for k < EE do
           let kk = i64.i32 k
           let V' = V[block_col*B : block_col*B + T*R, kk*B : kk*B + T*R] :> [T*R][T*R]f32
           -- let KVsh[:,0:T*R] = V'
           let KVsh = copy2shrPad T R KVsh V'
           let offsetO = kk*((T*T)*(R*R+1))
           -- do update to O
           let fOupd (tidd: i64) = #[unsafe]
             let tid = i32.i64 tidd
             let tidy = tid / TT
             let tidx = tid - tidy * TT
             let row_offset = tidy * RR
             let offset = offsetO + tidd*(R*R+1)
             let Omat = O[offset: offset + R*R] :> [R*R]f32
             let Omat = imapseq (iota R) 
                          (\ i -> imapseq (iota R) (\ j -> Omat[i*RR + j] * els[row_offset+i]))
             let Omat =
                 loop Omat for idx < BB do
                   let offs_K = i64.i32 (idx*(BB+1) + tidx * RR)
                   let indPout = tidy*TT + idx/RR in
                   imap2seq' Omat (iota R)
                     (\Orow i ->
                       let Pelm = #[unsafe] P[indPout][i*R + i64.i32 (idx%RR)] in
                       -- let Pelm = #[unsafe] P[indPout][i][idx%RR] in
                       imap2seq' Orow (iota R)
                         (\Oij j ->
                            Oij + Pelm * #[unsafe] KVsh[ offs_K + j]
                         )
                     ) 
             in  imapseq (iota (R*R+1))
                   (\ ind -> let (i,j) = (ind / RR, ind % RR)
                             in  if i == RR then 0 else Omat[i,j] / lis[row_offset + i]
                   )
           let Ok = map fOupd (iota (T*T))
           -- let O[offsetO : offsetO + (T*T)*(R*R)] = flatten (flatten Ok)
           let O  = accUpdateO T R k Ok O
           in  (O, KVsh)
   
       in  (O, mis, lis, maxs, sums, ess, els, KVsh)

  in O

entry mk_input (s:i64) (e:i64) (b:i64) : ([s][b][e*b]f32, [s][b][e*b]f32, [s][b][e*b]f32) =
  -- (n, d) = (s*b, e*b)
  let Q = replicate (e*b) 1.0 |> replicate b |> replicate s
  let K = replicate (e*b) 1.0 |> replicate b |> replicate s
  let V = replicate (e*b) 1.0 |> replicate b |> replicate s
  in  (Q, K, V)

def L2 [n] (xs: [n]f32) : f32 =
    map (\x -> x*x) xs |> reduce (+) 0.0 |> f32.sqrt  

--
-- ==
-- entry: main64
-- "Class 16384-64"  script input { (mk_input 256i64 1i64 64i64) }
-- output { 0.0f32 }
-- "Class 32768-64"  script input { (mk_input 512i64 1i64 64i64) }
-- output { 0.0f32 }

entry main64 [S] (Q0: [S][64][64]f32) (K0: [S][64][64]f32) (V0: [S][64][64]f32) =
  let Q = Q0 :> [S][16*4][1*16*4]f32
  let K = (flatten K0) :> [S*16*4][1*16*4]f32
  let V = (flatten V0) :> [S*16*4][1*16*4]f32
  let O = #[incremental_flattening(only_intra)]
          map (FlashAttention K V) Q |> opaque          
  let O_flat = flatten O
  in ( L2 O_flat ) - ( f32.sqrt (f32.i64 (S*64*64)) )
--  in  O


--
-- ==
-- entry: main128
-- "Class 8192-64 "  script input { (mk_input 256i64 4i64 32i64) }
-- output { 0.0f32 }
-- "Class 16384-64"  script input { (mk_input 512i64 4i64 32i64) }
-- output { 0.0f32 }

entry main128 [S] (Q0: [S][32][128]f32) (K0: [S][32][128]f32) (V0: [S][32][128]f32) =
  let Q = Q0 :> [S][8*4][4*8*4]f32
  let K = (flatten K0) :> [S*8*4][4*8*4]f32
  let V = (flatten V0) :> [S*8*4][4*8*4]f32
  let O = #[incremental_flattening(only_intra)]
          map (FlashAttention K V) Q |> opaque
  let O_flat = flatten O
  in ( L2 O_flat ) - ( f32.sqrt (f32.i64 (S*32*128)) )
--  in  O

----
---- ==
---- entry: main128R2
---- "Class 8192-64 "  script input { (mk_input 256i64 4i64 32i64) }
---- output { 0.0f32 }
---- "Class 16384-64"  script input { (mk_input 512i64 4i64 32i64) }
---- output { 0.0f32 }
--
--entry main128R2 [S] (Q0: [S][32][128]f32) (K0: [S][32][128]f32) (V0: [S][32][128]f32) =
--  let Q = Q0 :> [S][16*2][4*16*2]f32
--  let K = (flatten K0) :> [S*16*2][4*16*2]f32
--  let V = (flatten V0) :> [S*16*2][4*16*2]f32
--  let O = #[incremental_flattening(only_intra)]
--          map (FlashAttention K V) Q |> opaque
----  in  O
--  let O_flat = flatten O
--  in ( L2 O_flat ) - ( f32.sqrt (f32.i64 (S*32*128)) )
