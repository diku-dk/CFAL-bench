import "util"

------------------------------
--- Array Indexing Helpers ---
------------------------------

def getElm3d (arr:[][][]real) (i: i32) (j: i32) (k: i32) : real =
  #[unsafe] arr[i, j, k]

def getElmFlat [n] (arr:[n*n*n]real) (i: i32) (j: i32) (k: i32) : real =
  -- let n = i32.i64 n in #[unsafe] arr[ i*n*n + j*n + k ]
  #[unsafe] arr[ flatenInd (i,j,k) (i32.i64 n) (i32.i64 n) ]

def get8thElm3d (arr:[][][]real) (i: i32) (j: i32) (k: i32) : real =
  #[unsafe]
  if (i %% 2) + (j %% 2) + (k %% 2) == 3
  then arr[i//2,j//2,k//2]
  else 0

-------------------------------
--- SAC-like implementation ---
-------------------------------

def gen_weights (cs: [4]real) : [3][3][3]real =
  unroll_tabulate_3d 3 3 3 (\i j l -> #[unsafe] cs[i32.abs(i-1)+i32.abs(j-1)+i32.abs(l-1)])

def hood_3d 't (n: i64) (getElm: i32 -> i32 -> i32 -> t) i j l : [3][3][3]t =
  let nm1 = (i32.i64 n) - 1
  in  unroll_tabulate_3d 3 3 3
        (\a b c -> getElm ((i+a-1) & nm1) ((j+b-1) & nm1) ((l+c-1) & nm1) )
  
def relaxSAC (n: i64) (n': i64) (f: i32->i32) (getElm: i32->i32->i32->real)
             (ws: [4]real) : *[n][n][n]real =
  let flat_weights = gen_weights ws |> flatten_3d
  let f i j l =
    let hood = hood_3d n' getElm (f i) (f j) (f l)
    in #[sequential] #[unroll] sum (map2 (*) flat_weights (flatten_3d hood))
  in tabulate_3d' n n n f

def P [n] (a: [(n*2)*(n*2)*(n*2)]real) : *[n*n*n]real =
  flatten_3d (relaxSAC n (2*n) (\x->2*x+1) (getElmFlat a) [1/2, 1/4, 1/8, 1/16]) 
-- -- Or a bit more efficient and longer:
--  let f ijl =
--     let (i',j',l') = unflatInd ijl (i32.i64 n) (i32.i64 n)
--     let (i, j, l) = (2*i'+1, 2*j'+1, 2*l'+1)
--     let hood = hood_3d (2*n) (getElmFlat a) i j l
--     in  #[sequential] #[unroll] sum (map2 (*) (flatten_3d weights) (flatten_3d hood))
--  in map f (iota (n*n*n))

def coarse2fine [n] (z: [n][n][n]real) =
  tabulate_3d' (2*n) (2*n) (2*n) (get8thElm3d z)

def Qslow [n] (a: [n][n][n]real) =
  relaxSAC (2*n) (2*n) id (getElm3d (coarse2fine a)) [1,1/2,1/4,1/8]

-------------------------------
--- NAS-like implementation ---
-------------------------------

def relaxNAS (n: i64) (_n: i64) (_f: i32->i32) (getElm: i32->i32->i32->real)
             (ws: [4]real) : *[n][n][n]real =
  let nm1= (i32.i64 n) - 1
  let iterBody (i3: i32) (i2: i32) : [n]real =
      let f (i1: i32) = #[unsafe]
           (  getElm i3 ((i2-1) & nm1) i1 +
              getElm i3 ((i2+1) & nm1) i1 +
              getElm ((i3-1) & nm1) i2 i1 +
              getElm ((i3+1) & nm1) i2 i1
           ,  
              getElm ((i3-1) & nm1) ((i2-1) & nm1) i1 +
              getElm ((i3-1) & nm1) ((i2+1) & nm1) i1 +
              getElm ((i3+1) & nm1) ((i2-1) & nm1) i1 +
              getElm ((i3+1) & nm1) ((i2+1) & nm1) i1
           )
      let g u1s u2s (i1: i32) = #[unsafe]
           ( ws[0] * ( getElm i3 i2 i1 ) +
             ws[1] * ( getElm i3 i2 ((i1-1) & nm1) + getElm i3 i2 ((i1+1) & nm1) + u1s[i1] ) +
             ws[2] * ( u2s[i1] + u1s[(i1-1) & nm1] + u1s[(i1+1) & nm1] ) +
             ws[3] * ( u2s[(i1-1) & nm1] + u2s[(i1+1) & nm1] )
           )
      let (u1s, u2s) = unzip (tabulate' n f)
      in  tabulate' n (g u1s u2s)
  in  tabulateIntra_2d n n iterBody

-------------------------------
--- CORE ALGORITHMIC PIECES ---
-------------------------------

-- the type of the generic relaxation computational kernel:
type^ relaxT = (n: i64) -> i64 -> (i32 -> i32) -> 
               (i32->i32->i32->real) -> [4]real -> *[n][n][n]real

def Q [n] (relaxKer: relaxT) (a: [n][n][n]real) : [2*n][2*n][2*n]real =
  relaxKer (2*n) (2*n) id (get8thElm3d a) [1,1/2,1/4,1/8]

def mA [n] (relaxKer: relaxT) (a: [n][n][n]real) (v: [n][n][n]real) =
  relaxKer n n id (getElm3d a) [-8/3, 0, 1/6, 1/12] |> map2_3d (-) v

def pS [n] (relaxKer: relaxT) ws (a: [n][n][n]real) (v: [n][n][n]real) =
  relaxKer n n id (getElm3d a) ws |> map2_3d (+) v

def M [n] (relaxKer: relaxT) (wS: [4]real) (r: [n][n][n]real) : [n][n][n]real =
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
      let (off', m') = (off + m*m*m, m / 2)
      let r  = rss[off: off + m*m*m]
            |> sized ((m'*2)*(m'*2)*(m'*2))
      let rss[off': off' + m'*m'*m'] = P r
      in  (off', m', rss)

  -- base case of M
  let r1 = rss[off: off + m2*m2*m2]
        |> sized (2*2*2) |> unflatten_3d
  let z1 = pS relaxKer wS r1 (replicate_3d 2 0)

  -- loop back
  let (_, _, z) =
    loop (end, m, z) = (off, m2, z1)
    for _k < count do
      let (beg, m2) = (end - 8*m*m*m, m*2)
      let z' = (Q relaxKer z) :> [m2][m2][m2]real
      let r  = rss[beg : end] |> sized (m2*m2*m2) |> unflatten_3d
      let r' = mA relaxKer z' r 
      let z''= pS relaxKer wS r' z'
      in  (beg, m2, z'')
  -- treat the first case
  let z' = (Q relaxKer z) :> [n][n][n]real
  let r' = mA relaxKer z' r
  let z''= pS relaxKer wS r' z'
  in  z''

def L2 [n][m][q] (xsss: [n][m][q]real) : real =
  let s = flatten_3d xsss |> map (\x -> x*x) |> sum
  in  s / (int2Real (n*m*q)) |> sqrt

def mg [n] (relaxKer: relaxT) (iter: i64) (wS: [4]real) (v: [n][n][n]real) =
  let u = M relaxKer wS v
  let r = mA relaxKer u v
  let (_,r) =
    loop (u,r) for _i < iter-1 do
      let u' = M  relaxKer wS r |> map2_3d (+) u
      let r''= mA relaxKer u' v
      in  (u', r'')
  in  L2 r

entry mk_input n =
  let f i j k : real =
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

def mkSw (iter: i64) : [4]real = 
  let (s1, s2, s3) = 
    if iter == 4 then (-3/8,1/32,-1/64) else (-3/17,1/33,-1/61)
  in  [s1, s2, s3, 0]

entry mgNAS [n] (iter: i64) (v: [n][n][n]real) : real =
  mg relaxNAS iter (mkSw iter) v
  
entry mgSAC [n] (iter: i64) (v: [n][n][n]real) : real =
  mg relaxSAC iter (mkSw iter) v

-- Reference values: 0.2433365309e-5
--                   0.180056440132e-5    0.000001811585741f64
--                   0.5706732285740e-6
-- ==
-- entry: mgNAS mgSAC
-- "Class A" script input { (4i64, mk_input 256i64) }
-- output { 0.2433365309e-5 }
-- "Class B" script input { (20i64, mk_input 256i64) }
-- output { 0.180056440132e-5 }

-- "Class C" script input { (20i64, mk_input 512i64) }
-- output { 0.5706732285740e-6 }

-- "Class D" script input { (50i64, mk_input 1024i64) }
-- output { 0.1583275060440e-9 }
