import "util"
-- import "mg-nas-kers"

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

def hood_3d 't (n: i64) (getElm: i32 -> i32 -> i32 -> t) i j l : [3][3][3]t =
  let nm1 = (i32.i64 n) - 1
  in  unroll_tabulate_3d 3 3 3
        (\a b c -> getElm ((i+a-1) & nm1) ((j+b-1) & nm1) ((l+c-1) & nm1) )
  
def relaxSAC (n: i64) (n': i64) f (getElm: i32 -> i32 -> i32 -> real) (weights: [3][3][3]real) : *[n][n][n]real =
  let f i j l =
    let hood = hood_3d n' getElm (f i) (f j) (f l)
    in #[sequential] #[unroll] sum (map2 (*) (flatten_3d weights) (flatten_3d hood))
  in tabulate_3d' n n n f

def gen_weights (cs: [4]real) : [3][3][3]real =
  unroll_tabulate_3d 3 3 3 (\i j l -> #[unsafe] cs[i32.abs(i-1)+i32.abs(j-1)+i32.abs(l-1)])

def P [n] (a: [(n*2)*(n*2)*(n*2)]real) : *[n*n*n]real =
  let weights = gen_weights [1/2, 1/4, 1/8, 1/16]
  in  relaxSAC n (2*n) (\x->2*x+1) (getElmFlat a) weights |> flatten_3d
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
  relaxSAC (2*n) (2*n) id (getElm3d (coarse2fine a)) (gen_weights [1,1/2,1/4,1/8])

-------------------------------
--- NAS-like implementation ---
-------------------------------

def relaxNAS (n: i64) (getElm : i32 -> i32 -> i32 -> real) (a: [4]real) : [n][n][n]real =
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
           ( a[0] * ( getElm i3 i2 i1 ) +
             a[1] * ( getElm i3 i2 ((i1-1) & nm1) + getElm i3 i2 ((i1+1) & nm1) + u1s[i1] ) +
             a[2] * ( u2s[i1] + u1s[(i1-1) & nm1] + u1s[(i1+1) & nm1] ) +
             a[3] * ( u2s[(i1-1) & nm1] + u2s[(i1+1) & nm1] )
           )
      let (u1s, u2s) = unzip (tabulate' n f)
      in  tabulate' n (g u1s u2s)
  in  tabulateIntra_2d n n iterBody

-------------------------------
--- CORE ALGORITHMIC PIECES ---
-------------------------------

def Q [n] (withNAS: bool) (a: [n][n][n]real) : [2*n][2*n][2*n]real =
  if withNAS then relaxNAS (2*n) (get8thElm3d a) [1,1/2,1/4,1/8]
  else relaxSAC (2*n) (2*n) id (get8thElm3d a) (gen_weights [1,1/2,1/4,1/8])

def mA [n] (withNAS: bool) (v: [n][n][n]real) (a: [n][n][n]real) =
   map2_3d (-) v <|
   if withNAS then relaxNAS n (getElm3d a) [-8/3, 0, 1/6, 1/12]
   else relaxSAC n n id (getElm3d a) (gen_weights [-8/3, 0, 1/6, 1/12])

def S [n] (withNAS: bool) (ws, exp_ws) (a: [n][n][n]real) =
   if withNAS then relaxNAS n (getElm3d a) ws
   else relaxSAC n n id (getElm3d a) exp_ws

type S = ([4]real, [3][3][3]real)

def M [n] (withNAS: bool) (wS: S) (r: [n][n][n]real) : [n][n][n]real =
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
  let z1 = S withNAS wS r1

  -- loop back
  let (_, _, z) =
    loop (end, m, z) = (off, m2, z1)
    for _k < count do
      let m2 = m*2
      let z' = (Q withNAS z) :> [m2][m2][m2]real
      let beg = end - 8*m*m*m
      let r  = rss[beg : end] |> sized (m2*m2*m2) |> unflatten_3d
      let r' = mA withNAS r z'
      let z''= map2_3d (+) z' (S withNAS wS r')
      in  (beg, m2, z'')
  -- treat the first case
  let z' = (Q withNAS z) :> [n][n][n]real
  let r' = mA withNAS r z'
  let z''= map2_3d (+) z' (S withNAS wS r')
  in  z''

def L2 [n][m][q] (xsss: [n][m][q]real) : real =
  let s = flatten_3d xsss |> map (\x -> x*x) |> sum
  in  s / (int2Real (n*m*q)) |> sqrt

def mg [n] (withNAS: bool) (iter: i64) (wS: S) (v: [n][n][n]real) =
  let u = M withNAS wS v
  let r = mA withNAS v u
  let (_,r) =
    loop (u,r) for _i < iter-1 do
      let u' = M withNAS wS r |> map2_3d (+) u
      let r''= mA withNAS v u'
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

def mkSw (iter: i64) : S = 
  let (s1, s2, s3) =
      if iter == 4 then (-3/8,  1/32, -1/64) 
                   else (-3/17, 1/33, -1/61)
  let ws = [s1, s2, s3, 0]
  in  (ws, gen_weights ws)

entry mgNAS [n] (iter: i64) (v: [n][n][n]real) : real =
  mg true iter (mkSw iter) v
  
entry mgSAC [n] (iter: i64) (v: [n][n][n]real) : real =
  mg false iter (mkSw iter) v

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
