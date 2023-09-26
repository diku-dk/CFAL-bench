type real = f64

def sum  = f64.sum
def sqrt = f64.sqrt
def int2Real = f64.i64

def replicate_3d (n: i64) (v: real) : [n][n][n]real =
  replicate n v |> replicate n |> replicate n

def map2_3d f = map2 (map2 (map2 f))

def tabulate_3d' m n k f = tabulate_3d m n k (\i j k -> f (i32.i64 i) (i32.i64 j) (i32.i64 k))

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

def unroll_tabulate_3d n m l f =
  #[unroll]
  tabulate n (\a -> #[unroll]
                    tabulate m (\b -> #[unroll]
                                      tabulate l (\c -> f (i32.i64 a) (i32.i64 b) (i32.i64 c))))

def hood_3d [n] 't (arr: [n][n][n]t) i j l : [3][3][3]t =
  let nm1 = (i32.i64 n) - 1 in
  unroll_tabulate_3d 3 3 3 (\a b c -> #[unsafe] arr[(i+a-1) & nm1, (j+b-1) & nm1, (l+c-1) & nm1])

def relax [n] (input: [n][n][n]real) (weights: [3][3][3]real) : [n][n][n]real =
  let f i j l =
    let hood = hood_3d input i j l
    in #[sequential] #[unroll] sum (map2 (*) (flatten_3d weights) (flatten_3d hood))
  in tabulate_3d' n n n f

def hood_3dF [n] 't (arr: [n*n*n]t) i j l : [3][3][3]t =
  unroll_tabulate_3d 3 3 3
    (\a b c -> let n = i32.i64 n
               let ijl_rot = ( (i+a-1) & (n-1), (j+b-1) & (n-1), (l+c-1) & (n-1) )
               in  #[unsafe] arr[flatenInd ijl_rot n n] )

def relaxF [n] (input: [n*n*n]real) (weights: [3][3][3]real) : [n*n*n]real =
   let f ijl =
     let (i,j,l) = unflatInd ijl (i32.i64 n) (i32.i64 n)
     let hood = hood_3dF input i j l
     in  #[sequential] #[unroll] sum (map2 (*) (flatten_3d weights) (flatten_3d hood))
   in map f (iota (n*n*n))

def relaxFF [n] (input: [n*n*n]real) (weights: [3][3][3]real) : [n][n][n]real =
  let f i j l =
    let hood = hood_3dF input i j l
    in  #[sequential] #[unroll] sum (map2 (*) (flatten_3d weights) (flatten_3d hood))
  in (tabulate_3d' n n n f)


def gen_weights (cs: [4]real) : [3][3][3]real =
  unroll_tabulate_3d 3 3 3 (\i j l -> #[unsafe] cs[i32.abs(i-1)+i32.abs(j-1)+i32.abs(l-1)])

def coarse2fine [n] (z: [n][n][n]f64) =
  tabulate_3d' (2*n) (2*n) (2*n)
              (\i j k ->
                 #[unsafe]
                 if (i %% 2) + (j %% 2) + (k %% 2) == 3
                 then z[i//2,j//2,k//2]
                 else 0)

def fine2coarse [n][m][k] 't (r: [n*2][m*2][k*2]t) =
  r[1::2,1::2,1::2] :> [n][m][k]t
  -- tabulate_3d' n m k (\i j k -> #[unsafe]r[i*2+1, j*2+1, k*2+1]) 

def fine2coarseFF [n][m][k] 't (r: [(n*2)][(m*2)][(k*2)]t) : *[n*m*k]t =
  tabulate_3d' n m k 
    (\ i j l -> #[unsafe] r[i*2+1, j*2+1, l*2+1] ) |> flatten_3d
  -- r[1::2,1::2,1::2] :> [n][m][k]t
  
  
def fine2coarseF [n][m][k] 't (r: [(n*2)*(m*2)*(k*2)]t) : *[n*m*k]t =
  map (\ ijl ->
        let (m,k)    = (i32.i64 m, i32.i64 k)
        let (i,j,l)  = unflatInd ijl m k 
        let flat_ind = flatenInd (i*2+1, j*2+1, l*2+1) (m*2) (k*2)
        in  #[unsafe] r[flat_ind]
      ) (iota (n*m*k)) |> sized (n*m*k)
  -- r[0::2,0::2,0::2] :> [n][m][k]t
  
def P [n] (a: [(n*2)*(n*2)*(n*2)]real) : *[n*n*n]real =
  -- fine2coarseF (relaxF a (gen_weights [1/2, 1/4, 1/8, 1/16]))
  let weights = gen_weights [1/2, 1/4, 1/8, 1/16]

  let f ijl =
     let (i',j',l') = unflatInd ijl (i32.i64 n) (i32.i64 n)
     let (i, j, l) = (2*i'+1, 2*j'+1, 2*l'+1)
     let hood = hood_3dF a i j l
     in  #[sequential] #[unroll] sum (map2 (*) (flatten_3d weights) (flatten_3d hood))
  in map f (iota (n*n*n))
--  let f i' j' l' =
--    let (i, j, l) = (2*i'+1, 2*j'+1, 2*l'+1)
--    let hood = hood_3dF a i j l
--    in  #[sequential] #[unroll] sum (map2 (*) (flatten_3d weights) (flatten_3d hood))
--  in (tabulate_3d' n n n f) |> flatten_3d

def Q a = relax (coarse2fine a) (gen_weights [1,1/2,1/4,1/8])

type S = [3][3][3]f64

def Sa : S = gen_weights [-3/8, 1/32, -1/64, 0]

def Sb : S = gen_weights [-3/17, 1/33, -1/61, 0]

-- def Sa a = relax a (gen_weights [-3/8, 1/32, -1/64, 0])
-- def Sb a = relax a (gen_weights [-3/17, 1/33, -1/61, 0])

def A a = relax a (gen_weights [-8/3, 0, 1/6, 1/12])

def M [n] (S: S) (r: [n][n][n]real) : [n][n][n]real =
  -- compute the flat size of rss
  let (count, rs_flat_len, m0) =
    loop (count, len, m) = (0, 0, n/2) while m > 2 do
      (count+1, len + m*m*m, m/2)
  let rs_flat_len = rs_flat_len + m0 * m0 * m0
  -- allocate buffer size
  let rss = replicate rs_flat_len 0
  -- fill in rss
  let nd2 = n / 2
  let r_flat = ( flatten (flatten r) ) |> sized ( (nd2*2) * (nd2*2) * (nd2*2) )
  let rss[0: nd2*nd2*nd2] = P r_flat
  let (off, m2, rss) =
    loop (off, m, rss) = (0i64, n/2, rss)
    for _k < count do
      let off' = off + m*m*m
      let m' = m / 2
      let r  = rss[off: off + m*m*m] |> sized ((m'*2)*(m'*2)*(m'*2))
      let r' = P r
      let rss[off': off' + m'*m'*m'] = r'
      in  (off', m', rss)

  -- base case of M
  let r1 = rss[off: off + m2*m2*m2]
           |> sized (2*2*2) |> unflatten_3d
  let z1 = relax r1 S

  -- loop back
  let (_, _, z) =
    loop (end, m, z) = (off, m2, z1)
    for _k < count do
      let m2 = m*2
      let z' = (Q z) :> [m2][m2][m2]real
      let beg = end - 8*m*m*m
      let r  = rss[beg : end] |> sized (m2*m2*m2) |> unflatten_3d
      let r' = map2_3d (-) r (A z')
      let z''= map2_3d (+) z' (relax r' S)
      in  (beg, m2, z'')
  -- treat the first case
  let z' = (Q z) :> [n][n][n]real
  let r' = map2_3d (-) r (A z')
  let z''= map2_3d (+) z' (relax r' S)
  in  z''

def L2 [n][m][q] (xsss: [n][m][q]real) : real =
  -- sqrt(sum (map (\ x -> x * x) (flatten_3d xsss)) / int2Real (n*m*q))
  let s = flatten_3d xsss |> map (\x -> x*x) |> sum
  in  s / (int2Real (n*m*q)) |> sqrt

def mg [n] (iter: i64) (S: S) (v: [n][n][n]real) (u: [n][n][n]real) =
  let u =
    loop u for _i < iter do
      -- let r = v - A (u);
      let u' = A u
      let r  = map2_3d (-) v u'
      -- let u = u + M(r);
      let r' = M S r
      in  map2_3d (+) u r'
  in L2 (map2_3d (-) v (A u))

entry mk_input n =
  let f i j k : f64 =
    if any (==(i,j,k)) [(211,154,98),
                        (102,138,112),
                        (101,156,59),
                        (17,205,32),
                        (92,63,205),
                        (199,7,203),
                        (250,170,157),
                        (82,184,255),
                        (154,162,36),
                        (223,42,240)]
    then -1
    else if any (==(i,j,k)) [(57,120,167),
                             (5,118,175),
                             (176,246,164),
                             (45,194,234),
                             (212,7,248),
                             (115,123,207),
                             (202,83,209),
                             (203,18,198),
                             (243,172,14),
                             (54,209,40)]
    then 1
    else 0
  in tabulate_3d n n n f

entry main [n] (iter: i64) (v: [n][n][n]real) : real =
  let S = if iter == 4 then Sa else Sb
  in replicate_3d n 0 |> mg iter S v

-- Reference values: 0.2433365309e-5
--                   0.180056440132e-5
-- ==
-- entry: main
-- "Class A" script input { (4i64, mk_input 256i64) }
-- output { 0.2433365309e-5 }
-- "Class B" script input { (20i64, mk_input 256i64) }
-- output { 0.180056440132e-5 }
-- "Class D" script input { (50i64, mk_input 1024i64) }
-- output { 0.1583275060440e-9 }
