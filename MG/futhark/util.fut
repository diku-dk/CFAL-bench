type real = f64

def sum  = f64.sum
def sqrt = f64.sqrt
def int2Real = f64.i64

def replicate_3d (n: i64) (v: real) : [n][n][n]real =
  replicate n v |> replicate n |> replicate n

def map2_3d f = map2 (map2 (map2 f))

def tabulate' n f = tabulate n (\i -> f (i32.i64 i))

def tabulate_3d' m n k f = 
  tabulate_3d m n k (\i j k -> f (i32.i64 i) (i32.i64 j) (i32.i64 k))

def unroll_tabulate_3d n m l f =
  #[unroll]
  tabulate n (\a -> #[unroll]
                    tabulate m (\b -> #[unroll]
                                      tabulate l (\c -> f (i32.i64 a) (i32.i64 b) (i32.i64 c))))

def imapIntra as f =
  #[incremental_flattening(only_intra)] map f as

def tabulateIntra_2d n2 n1 f =
  tabulate' n2 (\i2 -> imapIntra (iota n1) (\i1 -> f i2 (i32.i64 i1)))
  
def unflatInd (ijl: i64) (n: i32) (k: i32) : (i32,i32,i32) =
   let ijl = i32.i64 ijl
   let nk = n*k
   let i  = ijl / nk
   let jl = ijl & (nk-1)
   let j = jl / k
   let l = jl & (k-1)
   in  (i, j, l)

def flatenInd (i: i32, j: i32, l:i32) (n: i32) (k: i32) : i64 =
  (i64.i32 i) * (i64.i32 (n*k)) + i64.i32 (j*k + l)
