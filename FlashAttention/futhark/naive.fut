type real = f32
def real_exp = f32.exp
def real_i64 = f32.i64
def real_max = f32.max
def real_lowest = f32.lowest
def real_sqrt = f32.sqrt


def imap xs f = map f xs

def setOne (n: i64) : [n]real = replicate n 1.0

def scale [m][n] (xss: [m][n]real) : [m][n]real =
  imap xss 
    (\xs -> let weight = reduce (+) 0 xs
            let weight'= 1.0 / weight
            in  map (* weight') xs
    )

def stabilize [m][n] (xss: [m][n]real) : [m][n]real =
  imap xss
    (\xs -> let maximum = reduce real_max real_lowest xs
            in  imap xs (\x -> x - maximum)
    )

def exp_mat [m][n] (xss: [m][n]real) : [m][n]real = 
  map (map real_exp) xss

def L2 [n] (xs: [n]real) : real =
    map (\x -> x*x) xs
    |> reduce (+) 0.0
    |> real_sqrt

def matmulT [m][n][k] (a: [m][k]real) (b: [n][k]real) : [m][n]real =
  imap a (\a_row -> imap b (\b_row -> map2 (*) a_row b_row |> reduce (+) 0.0) )

def matmul [m][n][k] (a: [m][k]real) (b: [k][n]real) : [m][n]real =
  matmulT a (transpose b)

def oneIter [d][m] (K: [m*d][d]real) (V: [m*d][d]real) (Qi: [d][d]real) : [d][d]real =
  let P_block = matmulT Qi K  -- : [d][m*d]real 
  let P_block = P_block |> stabilize |> exp_mat |> scale
  in  matmul P_block V

def FlashAttention [d][m] 
        (Q: [m][d][d]real) 
        (K: [m*d][d]real) 
        (V: [m*d][d]real) 
      : [m][d][d]real =
  map (oneIter K V) Q
  
--
-- ==
-- input { 65536i64 512i64 }
-- output { 0f32 }

entry main (m:i64) (d:i64) : real =
  let Q = replicate d 1.0 |> replicate d |> replicate m
  let K = replicate d 1.0 |> replicate (m*d)
  let V = replicate d 1.0 |> replicate (m*d)
  
  let O = FlashAttention Q K V
   
  let O_flat = flatten (flatten O)
  in ( L2 O_flat ) - (real_sqrt (real_i64 (m*d*d)))
