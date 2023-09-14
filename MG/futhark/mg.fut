-- Not done; only the relaxation for now.

def unroll_tabulate_3d n m l f =
  #[unroll]
  tabulate n (\a -> #[unroll]
                    tabulate m (\b -> #[unroll]
                                      tabulate l (\c -> f a b c)))

def hood_3d [n] 't (arr: [n][n][n]t) i j l : [3][3][3]t =
  unroll_tabulate_3d 3 3 3 (\a b c -> #[unsafe] arr[(i+a-1)%n,(j+b-1)%n,(l+c-1)%n])

def hoods_3d [n] 't (arr: [n][n][n]t) : [n][n][n][3][3][3]t =
  tabulate_3d n n n (hood_3d arr)

entry relax [n] (input: [n][n][n]f64) (weights: [3][3][3]f64) : [n][n][n]f64 =
  let f i j l =
    let hood = hood_3d input i j l
    in #[sequential] #[unroll] f64.sum (map2 (*) (flatten_3d weights) (flatten_3d hood))
  in tabulate_3d n n n f

def gen_weights (cs: [4]f64) =
  tabulate_3d 3 3 3 (\i j l -> cs[i64.abs(i-1)+i64.abs(j-1)+i64.abs(l-1)])

def dup = replicate 2 >-> transpose >-> flatten

-- FIXME: is this right?
def coarse2fine z =
  z
  |> map (map dup)
  |> map dup
  |> dup

def fine2coarse [n][m][k] 't (r: [n*2][m*2][k*2]t) =
  r[0::2,0::2,0::2] :> [n][m][k]t

def P a = fine2coarse (relax a (gen_weights [1/2, 1/4, 1/8, 1/16]))

def Q a = relax (coarse2fine a) (gen_weights [1,1/2,1/4,1/8])

def Sa a = relax a (gen_weights [-3/8, 1/32, -1/64, 0])

def Sb a = relax a (gen_weights [-3/17, 1/33, -1/61, 0])
