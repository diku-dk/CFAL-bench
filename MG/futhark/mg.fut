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

