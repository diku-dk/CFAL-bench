type real = f32
def real_exp = f32.exp
def real_i64 = f32.i64
def real_max = f32.max
def real_lowest = f32.lowest
def real_sqrt = f32.sqrt

def imap xs f = map f xs

def stabilize [m][n] (xss: [m][n]real) : [m][n]real =
  imap xss
    (\xs -> let maximum = reduce real_max real_lowest xs
            in  imap xs (\x -> x - maximum)
    )

def exp_mat [m][n] (xss: [m][n]real) : [m][n]real = 
  map (map real_exp) xss

def scale [m][n] (xss: [m][n]real) : [m][n]real =
  imap xss 
    (\xs -> let weight = reduce (+) 0 xs
            let weight'= 1.0 / weight
            in  map (* weight') xs
    )

def matmulT [m][n][k] (a: [m][k]real) (b: [n][k]real) : [m][n]real =
  imap a (\a_row -> imap b (\b_row -> map2 (*) a_row b_row |> reduce (+) 0.0) )

def matmul [m][n][k] (a: [m][k]real) (b: [k][n]real) : [m][n]real =
  matmulT a (transpose b)

def oneIter [d][m] (K: [m*d][d]real) (V: [m*d][d]real) (Qi: [d][d]real) : [d][d]real =
  let P_block = matmulT Qi K |> opaque -- : [d][m*d]real 
  let P_block = P_block |> stabilize |> exp_mat |> scale 
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
  
--
-- entry: main1
-- input { 65536i64 512i64 }
-- output { 0f32 }

entry main1 (m:i64) (d:i64) : real =
  let Q = replicate d 1.0 |> replicate d |> replicate m
  let K = replicate d 1.0 |> replicate (m*d)
  let V = replicate d 1.0 |> replicate (m*d)
  
  let O = FlashAttention Q K V  |> opaque
   
  let O_flat = flatten (flatten O)
  in ( L2 O_flat ) - (real_sqrt (real_i64 (m*d*d)))
  
  
entry mk_input (m:i64) (d:i64) : ([m][d][d]real, [m*d][d]real, [m*d][d]real) =
  let Q = replicate d 1.0 |> replicate d |> replicate m
  let K = replicate d 1.0 |> replicate (m*d)
  let V = replicate d 1.0 |> replicate (m*d)
  in  (Q, K, V)

--
-- ==
-- entry: main2
-- "Class 64-128" script input { (mk_input 64i64 128i64) }
-- output { 0.0f32 }
-- "Class 128-128" script input { (mk_input 128i64 128i64) }
-- output { 0.0f32 }
-- "Class 128-64" script input { (mk_input 128i64 64i64) }
-- output { 0.0f32 }
-- "Class 256-64" script input { (mk_input 256i64 64i64) }
-- output { 0.0f32 }

entry main2 [m][d] (Q: [m][d][d]real) (K: [m*d][d]real) (V: [m*d][d]real) : real =
  let O = FlashAttention Q K V
  let O_flat = flatten (flatten O)
  in ( L2 O_flat ) - (real_sqrt (real_i64 (m*d*d)))
