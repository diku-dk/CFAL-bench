type real = f64

def sum  = f64.sum
def sqrt = f64.sqrt
def int2Real = f64.i64

def replicate_3d (n: i64) (v: real) : [n][n][n]real =
  replicate n v |> replicate n |> replicate n

def map2_3d f = map2 (map2 (map2 f))

def unroll_tabulate_3d n m l f =
  #[unroll]
  tabulate n (\a -> #[unroll]
                    tabulate m (\b -> #[unroll]
                                      tabulate l (\c -> f a b c)))

def hood_3d [n] 't (arr: [n][n][n]t) i j l : [3][3][3]t =
  unroll_tabulate_3d 3 3 3 (\a b c -> #[unsafe] arr[(i+a-1)%n,(j+b-1)%n,(l+c-1)%n])

def relax [n] (input: [n][n][n]real) (weights: [3][3][3]real) : [n][n][n]real =
  let f i j l =
    let hood = hood_3d input i j l
    in #[sequential] #[unroll] sum (map2 (*) (flatten_3d weights) (flatten_3d hood))
  in tabulate_3d n n n f

def gen_weights (cs: [4]real) : [3][3][3]real =
  unroll_tabulate_3d 3 3 3 (\i j l -> cs[i64.abs(i-1)+i64.abs(j-1)+i64.abs(l-1)])

def coarse2fine [n] (z: [n][n][n]f64) =
  tabulate_3d (2*n) (2*n) (2*n)
              (\i j k ->
                 #[unsafe]
                 if (i %% 2) + (j %% 2) + (k %% 2) == 3
                 then z[i//2,j//2,k//2]
                 else 0)

def fine2coarse [n][m][k] 't (r: [n*2][m*2][k*2]t) =
  r[1::2,1::2,1::2] :> [n][m][k]t

def P a = fine2coarse (relax a (gen_weights [1/2, 1/4, 1/8, 1/16]))

def Q a = relax (coarse2fine a) (gen_weights [1,1/2,1/4,1/8])

type S = [3][3][3]f64

def Sa : S = gen_weights [-3/8, 1/32, -1/64, 0]

def Sb : S = gen_weights [-3/17, 1/33, -1/61, 0]

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
  let rss[0: nd2*nd2*nd2] = P (r :> [nd2*2][nd2*2][nd2*2]real) |> flatten_3d
  let (off, m2, rss) =
    loop (off, m, rss) = (0i64, n/2, rss)
    for _k < count do
    let r  = rss[off: off + m*m*m]
             |> sized (m*m*m) |> unflatten_3d -- (n/2) (n/2) (n/2)
      let m' = m / 2
      let off' = off + m*m*m
      let r' = P (r :> [m'*2][m'*2][m'*2]real) |> flatten_3d
      let rss[off': off' + m'*m'*m'] = copy r'
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
  f64.sqrt(f64.sum (map (**2) (flatten_3d xsss)) / f64.i64 (n*m*q))

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
-- "Class C" script input { (20i64, mk_input 512i64) }
-- output { 0.5706732285740e-6 }
