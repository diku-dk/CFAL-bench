type real = f32
def real_exp = f32.exp
def real_i64 = f32.i64
def real_max = f32.max
def real_lowest = f32.lowest
def real_sqrt = f32.sqrt
def real_zero = 0f32

def map4Intra f =
  #[incremental_flattening(only_intra)] map4 f

def imap xs f = map f xs

def imap2 xs ys f = map2 f xs ys

def imap4 as bs cs ds f = map4 f as bs cs ds

def tab_seq q f = #[sequential] map f (iota q)

def matmulT [m][n][k] (a: [m][k]real) (b: [n][k]real) : [m][n]real =
  imap a (\a_row -> imap b (\b_row -> map2 (*) a_row b_row |> reduce (+) 0.0) )

def dotprod [d] (as: [d]real) (bs: [d]real) : real =
  #[sequential] map2 (*) as bs |> 
  #[sequential] reduce (+) 0

------------------------
--- In-Progress Work ---
------------------------

def FlashAttention [d][N]
        (M : i64)
        -- (O: [N][d]real)
        (Q: [N][d]real) 
        (K: [N][d]real) 
        (V: [N][d]real) 
      : ([N]real, [N]real, [N][d]real) =
  
  let Bc = M / (4 * d)
  let Br = i64.min d Bc   -- d < Bc ? d : Bc;
  let Tr = N / Br
  let Tc = N / Bc
  
  let ms = replicate Tr (replicate Br real_lowest)
  let ls = replicate Tr (replicate Br real_zero)
  -- let Os_new = replicate (Tr*Br) (replicate d real_zero))
  -- let O  = (O :> [Tr*Br][d]real) |> unflatten
  let O = replicate d real_zero |> replicate Br |> replicate Tr

  -- j = 0 .. Tc-1 must be a sequential loop 
  -- dictated by the computation of mi and li
  let (ms', ls', O') =
    loop (ms, ls, O)
    for j < Tc do
      let f (mi: [Br]real) (li: [Br]real) (Oi: [Br][d]real) i : ([Br]real, [Br]real, [Br][d]real) = #[unsafe]
        -- ToDo: copy Kj, Vj, Oi, Qi to shared memory
        -- let Oi = O[i*Br : (i+1)*Br] :> [Br][d]real
        let Qi = Q[i*Br : (i+1)*Br] :> [Br][d]real

        -- can Kj and Vj be used from LL$ instead of shared memory
        let Kj = K[j*Bc : (j+1)*Bc] :> [Bc][d]real
        let Vj = V[j*Bc : (j+1)*Bc] :> [Bc][d]real
        
        let Pijs = tabulate_2d Br Bc 
                    (\ ii jj -> #[unsafe] dotprod Qi[ii, 0:d] Kj[jj, 0:d] ) |> opaque
        
        let (maxs, Pijs) = imap Pijs
            (\ Pij -> let max = reduce real_max real_lowest Pij
                      let xs  = map (\x -> real_exp (x - max)) Pij
                      in  (max, xs)
            ) |> unzip |> opaque
            
        let sums = imap Pijs (reduce (+) 0) |> opaque
        
        let (mi, li, eijs, elis) = imap4 mi li maxs sums
          (\ mii lii max sum ->
            let mi_new = real_max max mii
            let eij = real_exp (max - mi_new)
            let eli = lii * (real_exp (mii - mi_new))
            let li_new = eli + sum * eij
            in  (mi_new, li_new, eij, eli)
          ) |> unzip4 |> opaque
        
--        let (mi, li, eijs, elis, Pijs) = imap4 mi li Pijs (iota Br)
--          (\ mii lii Pij _ii ->
--            let max = reduce real_max real_lowest Pij
--            let xs  = imap Pij (\x -> real_exp (x - max))
--            let sum = reduce (+) 0 xs
--            let mi_new = real_max max mii
--            let eij = real_exp (max - mi_new)
--            let eli = lii * (real_exp (mii - mi_new))
--            let li_new = eli + sum * eij
--            in  (mi_new, li_new, eij, eli, xs)
--          ) |> unzip5 |> opaque
          
        let Oi_new =
          loop (Oi_new) = copy Oi
          for k < d/Bc do #[unsafe]
            let mkOi li_new eij eli Oii Pij = 
              imap2 (Oii[k*Bc : (k+1)*Bc] :> [Bc]real) Pij
                (\ elmOi pij -> elmOi + pij * li_new)
            let acc_chunk = map5 mkOi maxs eijs elis Oi Pijs
            -- BUG: replacing `maxs` with `mi` in the line above explodes
            -- the shared-emmory usage from some 6K to 532K 
            -- (for no apparent good reason as `mi` has small size `Br`)

--              imap2 (Oii[k*Bc : (k+1)*Bc] :> [Bc]real) (iota Bc)
--                    (\ elmOi kk -> 
--                      let acc = #[sequential] map2 (\ pij vjk -> eij*pij*vjk) Pij (Vj[:, k*Bc + kk])
--                             |> #[sequential] reduce (+) 0
--                      in #[unsafe] (elmOi * eli + acc) / li_new
--                    ) 
--            let acc_chunk = map5 mkOi li eijs elis Oi Pijs
            let Oi_new[:, k*Bc : (k+1)*Bc] = acc_chunk
            in  Oi_new
            -- 
        in  (mi, li, Oi_new)
      
      in  map4Intra f ms ls O (iota Tr) |> unzip3
  --
  let ms = (flatten ms') :> [N]real
  let ls = (flatten ls') :> [N]real
  let O''= (flatten  O') :> [N][d]real
  in  (ms, ls, O'')

entry query_sizes [n][d] (M: i64) (_Q: [n][d]real) (_K: [n][d]real) (_V: [n][d]real) : (i64,i64,i64,i64,i64,i64,i64) =
  let Bc = M / (4 * d)
  let Br = i64.min d Bc   -- d < Bc ? d : Bc;
  let Tr = n / Br
  let Tc = n / Bc
  in  (M, n, d, Bc, Br, Tr, Tc)

entry mk_input (n:i64) (d:i64) : (i64, [n][d]real, [n][d]real, [n][d]real) =
  let tile = 16
  let M = tile * 4 * d
  let Q = replicate d 1.0 |> replicate n
  let K = replicate d 1.0 |> replicate n
  let V = replicate d 1.0 |> replicate n
  in  (M, Q, K, V)


def L2 [n] (xs: [n]real) : real =
    map (\x -> x*x) xs
    |> reduce (+) 0.0
    |> real_sqrt

--
-- ==
-- entry: flashalg1
-- "Class 8192-128" script input { (mk_input 8192i64 128i64) }
-- output { 0.0f32 }
-- "Class 16384-128" script input { (mk_input 16384i64 128i64) }
-- output { 0.0f32 }
-- "Class 8192-64" script input { (mk_input 8192i64 64i64) }
-- output { 0.0f32 }
-- "Class 16384-64" script input { (mk_input 16384i64 64i64) }
-- output { 0.0f32 }

entry flashalg1 [n][d] (M: i64) (Q: [n][d]real) (K: [n][d]real) (V: [n][d]real) : real =
  let (_, _, O) = FlashAttention M Q K V
  let O_flat = flatten O
  in ( L2 O_flat ) - (real_sqrt (real_i64 (n*d)))

entry debug [n][d] (M: i64) (Q: [n][d]real) (K: [n][d]real) (V: [n][d]real) =
  let (ms, ls, O) = FlashAttention M Q K V
  in  O -- (ms, ls)

