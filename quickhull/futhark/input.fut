import "lib/github.com/diku-dk/segmented/segmented"

type char = u8
type string [n] = [n]char

def dtoi (c: u8): i32 = i32.u8 c - '0'

def is_digit (c: u8) = c >= '0' && c <= '9'

def atoi [n] (s: string[n]): i32 =
  let (sign,s) = if n > 0 && s[0] == '-' then (-1,drop 1 s) else (1,s)
  in sign * (loop (acc,i) = (0,0) while i < length s do
               if is_digit s[i]
               then (acc * 10 + dtoi s[i], i+1)
               else (acc, n)).0

def f &&& g = \x -> (f x, g x)

module words : {
  type word [p]
  val words [n] : [n]char -> ?[p].(word [p] -> ?[m].[m]char, ?[k].[k](word [p]))
} = {
  def is_space (x: char) = x == ' ' || x == '\n'
  def isnt_space x = !(is_space x)

  type word [p] = ([p](), i64, i64)

  def words [n] (s: [n]char) =
    (\(_, i, k) -> #[unsafe] s[i:i+k],
     segmented_scan (+) 0 (map is_space s) (map (isnt_space >-> i64.bool) s)
     |> (id &&& rotate 1)
     |> uncurry zip
     |> zip (indices s)
     |> filter (\(_,(x,y)) -> x > y)
     |> map (\(i,(x,_)) -> ([],i-x+1,x)))
}

def points_from_string (s: []u8) : [][2]f64 =
  let (get, ws) = words.words s
  in tabulate (length ws / 2) (\i -> [f64.i32 (atoi (get ws[i*2])),
                                      f64.i32 (atoi (get ws[i*2+1]))])
