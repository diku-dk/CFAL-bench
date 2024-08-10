type real = f32
def real_exp = f32.exp
def real_i64 = f32.i64
def real_max = f32.max
def real_lowest = f32.lowest
def real_sqrt = f32.sqrt

def imap xs f = map f xs
def imap3 xs ys zs f = map3 f xs ys zs

def chunk : i64 = 1024
def fseq  : i64 = 32

def copy2shr [n] (xs: [n]real) : *[n]real = #[unsafe]
  let xs' = copy xs
  in  if opaque(true) then xs'
      else xs' with [0] = 0f32

def reduceEffSeq [g] (f: real -> real) (bop: real->real->real) (ne: real) (xs: [g*fseq]real) : real = #[unsafe]
  let redPerThd (tid: i64) =
    loop r = ne for i < fseq do
        bop r (f xs[i*g + tid])
  -- per-thread reduce
  let rs = map redPerThd (iota g)
  in  reduce_comm bop ne rs

def softmaxChunkML (q: i64) (xs_glb: [q*chunk]real) : (real,real) = #[unsafe]
  let g = chunk/fseq in
  loop (mi_old : real, li_old : real) = (real_lowest, 0.0)
  for i < q do
    let xs = copy2shr ( xs_glb[i*chunk: i*chunk + chunk] :> [g*fseq]real )
    let xs = xs
    --
    let maxi = reduceEffSeq id real_max real_lowest xs
    let sumi = reduceEffSeq (\x -> real_exp (x - maxi)) (+) 0.0 xs
    --
    let mi_new = real_max mi_old maxi
    let eij = real_exp (maxi - mi_new)
    let eli = li_old * (f32.exp (mi_old - mi_new))
    let li_new = eli + sumi * eij
    in  (mi_new, li_new)
    -- this saves one f32.exp operation:
--    let exp_term = real_exp (mi_old - maxi)
--    in  if mi_old < maxi
--        then ( maxi,   li_old * exp_term + sumi )
--        else ( mi_old, li_old + sumi / exp_term )  

def softmaxOnline [m][n] (xss: [m][n]real) : [m][n]real = #[unsafe]
  let q = assert (n % chunk == 0) (n / chunk)
  let (mis, lis) = 
          #[incremental_flattening(only_intra)]
          map (softmaxChunkML q) (xss :> [m][q*chunk]real)
      |> unzip |> opaque
  in  imap3 xss mis lis
        (\ xs mi li ->
          map (\ x -> real_exp (x - mi) / li ) xs
        ) |> opaque

def softmax [m][n] (xss: [m][n]real) : [m][n]real =
  let f xs =
    let max = reduce real_max real_lowest xs
    let xs' = map (\x -> real_exp (x - max)) xs
    let weight = 1.0 / reduce (+) 0.0 xs'
    in  map (* weight) xs'
  in map f xss

def matmulT [m][n][k] (a: [m][k]real) (b: [n][k]real) : [m][n]real =
  imap a (\a_row -> imap b (\b_row -> map2 (*) a_row b_row |> reduce (+) 0.0) )

def matmul [m][n][k] (a: [m][k]real) (b: [k][n]real) : [m][n]real =
  matmulT a (transpose b)

def oneIter [d][m] (K: [m*d][d]real) (V: [m*d][d]real) (Qi: [d][d]real) : [d][d]real =
  let P_block = matmulT Qi K |> opaque -- : [d][m*d]real 
  -- let P_block = softmax P_block
  let P_block = softmaxOnline P_block
  in  matmul P_block V      -- : [d][d]real

def FlashAttention [d][m] 
        (Q: [m][d][d]real) 
        (K: [m*d][d]real) 
        (V: [m*d][d]real) 
      : [m][d][d]real =
  map (oneIter K V) Q

def L2 [n] (xs: [n]real) : real =
    map (\x -> x*x) xs
    |> reduce (+) 0.0
    |> real_sqrt  
  
entry mk_input (m:i64) (d:i64) : ([m][d][d]real, [m*d][d]real, [m*d][d]real) =
  let Q = replicate d 1.0 |> replicate d |> replicate m
  let K = replicate d 1.0 |> replicate (m*d)
  let V = replicate d 1.0 |> replicate (m*d)
  in  (Q, K, V)

--
-- ==
-- entry: main64
-- "Class 16384-64 " script input { (mk_input 256i64 64i64) }
-- "Class 32768-64 " script input { (mk_input 512i64 64i64) }

entry main64 [m] (Q: [m][64][64]real) (K: [m*64][64]real) (V: [m*64][64]real) =
  FlashAttention Q K V

--
-- ==
-- entry: main128
-- "Class 8192-128 " script input { (mk_input 64i64 128i64) }
-- "Class 16384-128" script input { (mk_input 128i64 128i64) }

entry main128 [m] (Q: [m][128][128]real) (K: [m*128][128]real) (V: [m*128][128]real) =
  FlashAttention Q K V

--
-- ==
-- entry: validate
-- "Class 16384-64 " script input { (mk_input 256i64 64i64) }
-- output { 0.0f32 }
-- "Class 32768-64 " script input { (mk_input 512i64 64i64) }
-- output { 0.0f32 }
-- "Class 8192-128 " script input { (mk_input 64i64 128i64) }
-- output { 0.0f32 }
-- "Class 16384-128" script input { (mk_input 128i64 128i64) }
-- output { 0.0f32 }

entry validate [m][d] (Q: [m][d][d]real) (K: [m*d][d]real) (V: [m*d][d]real) : real =
  let O = FlashAttention Q K V
  let O_flat = flatten (flatten O)
  in ( L2 O_flat ) - (real_sqrt (real_i64 (m*d*d)))
  -- Denoting with N = m*d, 
  -- THE NUMBER OF FLOPS IS: 4 * d * N * N
  -- ALSO, Datasets are named "Class N-d"

