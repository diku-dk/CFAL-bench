type real = f32
def real_exp = f32.exp
def real_i64 = f32.i64
def real_max = f32.max
def real_lowest = f32.lowest
def real_sqrt = f32.sqrt
def real_zero = 0f32

def map4Intra f =
  #[incremental_flattening(only_intra)] map4 f

def imap xs f = map f xs

def imap2 xs ys f = map2 f xs ys

def imap4 as bs cs ds f = map4 f as bs cs ds

def matmulT [m][n][k] (a: [m][k]real) (b: [n][k]real) : [m][n]real =
  imap a (\a_row -> imap b (\b_row -> map2 (*) a_row b_row |> reduce (+) 0.0) )

def dotprod [d] (as: [d]real) (bs: [d]real) : real =
  #[sequential] map2 (*) as bs |> 
  #[sequential] reduce (+) 0

------------------------
--- In-Progress Work ---
------------------------



-- N = m*d   4*d | M?


def FlashAttention [d][N]
        (M : i64)
        -- (O: [N][d]real)
        (Q: [N][d]real) 
        (K: [N][d]real) 
        (V: [N][d]real) 
      : ([N]real, [N]real, [N][d]real) =
  
  let Bc = M / (4 * d)
  let Br = i64.min d Bc   -- d < Bc ? d : Bc;
  let Tr = N / Br
  let Tc = N / Bc
  
  let ms = replicate Tr (replicate Br real_lowest)
  let ls = replicate Tr (replicate Br real_zero)
  -- let Os_new = replicate (Tr*Br) (replicate d real_zero))
  -- let O  = (O :> [Tr*Br][d]real) |> unflatten
  let O = replicate d real_zero |> replicate Br |> replicate Tr

  -- j = 0 .. Tc-1 must be a sequential loop 
  -- dictated by the computation of mi and li
  let (ms', ls', O') =
    loop (ms, ls, O)
    for j < Tc do
      let f (mi: [Br]real) (li: [Br]real) (Oi: [Br][d]real) i =
        -- ToDo: copy Kj, Vj, Oi, Qi to shared memory
        -- let Oi = O[i*Br : (i+1)*Br] :> [Br][d]real
        let Qi = Q[i*Br : (i+1)*Br] :> [Br][d]real

        -- can Kj and Vj be used from LL$ instead of shared memory
        let Kj = K[j*Bc : (j+1)*Bc] :> [Bc][d]real
        let Vj = V[j*Bc : (j+1)*Bc] :> [Bc][d]real
        
        in
         imap4 mi li Oi (iota Br)
          (\ mii lii Oii ii -> 
            let Pij = tabulate Bc (\jj -> dotprod Qi[ii, 0:d] Kj[jj, 0:d]) |> opaque
            let max = reduce real_max real_lowest Pij
            let xs  = imap Pij (\x -> real_exp (x - max))  -- *x = exp(*x - max)
            let sum = reduce (+) 0 xs
            let mi_new = real_max max mii
            let eij = real_exp (max - mi_new)
            let eli = lii * (real_exp (mii - mi_new))
            let li_new = eli + sum * eij
            --
            let Oii_new =
              loop (Oii_new) = copy Oii
              for k < d/Bc do #[unsafe]
                let acc_chunk = imap2 Oii[k*Bc : (k+1)*Bc] (iota ((k+1)*Bc - k*Bc))
                  (\ elmOi kk -> 
                      let acc = #[sequential] map2 (\ pij vjk -> eij*pij*vjk) xs (Vj[:, k*Bc + kk])  -- was Pij instead of xs
                             |> #[sequential] reduce (+) 0
                      in #[unsafe] (elmOi * eli + acc) / li_new
                  )
                let Oii_new[k*Bc : (k+1)*Bc] = acc_chunk
                in  Oii_new
            -- 
--------------------------------------------------------------------------------
--              for (int kk = 0; kk < d; kk++)  // a map-like update that must be mapped on parallel dimension Bc
--                      Oi[ii * d + kk] *= eli;
--              for (int jj = 0; jj < Bc; jj++) {  // Sequential!!!
--                      float x = eij * Pij[jj];
--
--                      for (int kk = 0; kk < d; kk++) { // Parallel (mapped on Bc)
--                              int ik = ii * d + kk;
--                              int jk = jj * d + kk;
--                              Oi[ik] += x * Vj[jk];  // O[i*Br+ii, kk] += eij * Pij[jj] * V[j*Bc+jj, kk]
--                      }
--              }
--
--              for (int kk = 0; kk < d; kk++)  // mapped on Bc again
--                      Oi[ii * d + kk] /= li_new;
----------------------------------------------------------------------------------
            in  (mi_new, li_new, Oii_new)
          )
      in  map4Intra f ms ls O (iota Tr) |> map unzip3 |> unzip3
  --
  let ms = (flatten ms') :> [N]real
  let ls = (flatten ls') :> [N]real
  let O''= (flatten  O') :> [N][d]real
  in  (ms, ls, O'')

entry query_sizes [n][d] (M: i64) (_Q: [n][d]real) (_K: [n][d]real) (_V: [n][d]real) : (i64,i64,i64,i64,i64,i64,i64) =
  let Bc = M / (4 * d)
  let Br = i64.min d Bc   -- d < Bc ? d : Bc;
  let Tr = n / Br
  let Tc = n / Bc
  in  (M, n, d, Bc, Br, Tr, Tc)

entry mk_input (n:i64) (d:i64) : (i64, [n][d]real, [n][d]real, [n][d]real) =
  let tile = 16
  let M = tile * 4 * d
  let Q = replicate d 1.0 |> replicate n
  let K = replicate d 1.0 |> replicate n
  let V = replicate d 1.0 |> replicate n
  in  (M, Q, K, V)


def L2 [n] (xs: [n]real) : real =
    map (\x -> x*x) xs
    |> reduce (+) 0.0
    |> real_sqrt

--
-- ==
-- entry: flashalg1
-- "Class 8192-128" script input { (mk_input 8192i64 128i64) }
-- output { 0.0f32 }
-- "Class 16384-128" script input { (mk_input 16384i64 128i64) }
-- output { 0.0f32 }
-- "Class 8192-64" script input { (mk_input 8192i64 64i64) }
-- output { 0.0f32 }
-- "Class 16384-64" script input { (mk_input 16384i64 64i64) }
-- output { 0.0f32 }

entry flashalg1 [n][d] (M: i64) (Q: [n][d]real) (K: [n][d]real) (V: [n][d]real) : real =
  let (_, _, O) = FlashAttention M Q K V
  let O_flat = flatten O
  in ( L2 O_flat ) - (real_sqrt (real_i64 (n*d)))

entry debug [n][d] (M: i64) (Q: [n][d]real) (K: [n][d]real) (V: [n][d]real) =
  let (ms, ls, O) = FlashAttention M Q K V
  in  O -- (ms, ls)


-----------------------------
--- The C implementation: ---
-----------------------------

--
-- int flash_attention(float *O, float *Q, float *K, float *V, int N, int d, int M)
-- {
--	float *l, *m, *Pij;
--	int Br, Bc, Tr, Tc;
--	
--	Bc = M / (4 * d);
--	Br = d < Bc ? d : Bc;   // Br = min (d, Bc)
--	Tr = N / Br;
--	Tc = N / Bc;
--	
--	if ((l = calloc(N, sizeof(float))) == NULL)
--		return 1;
--	
--	if ((m = calloc(N, sizeof(float))) == NULL)
--		return 1;
--	
--	if ((Pij = calloc(Bc, sizeof(float))) == NULL)
--		return 1;
--	
--	for (int i = 0; i < N; i++)
--		m[i] = -INFINITY;
--	
--	for (int j = 0; j < Tc; j++) {
--		float *Kj, *Vj;
--		
--		Kj = K + j * Bc * d;
--		Vj = V + j * Bc * d;
--		
--		for (int i = 0; i < Tr; i++) {
--			float *Oi, *Qi, *mi, *li;
--			
--			Oi = O + i * Br * d;
--			Qi = Q + i * Br * d;
--			mi = m + i * Br;
--			li = l + i * Br;
--			
--			// READS:     Q[i*Br : i*(Br+1)]   K[j*Bc: j*(Bc+1)]
--			// COMPUTES:  Pij[0:Br][0:Bc]
--			for (int ii = 0; ii < Br; ii++) {  // par?
--				float sum, max, li_new, mi_new, eij, eli;
--				
--				max = -INFINITY;
--
--				for (int jj = 0; jj < Bc; jj++) { // par?
--					sum = 0;
--					
--					for (int kk = 0; kk < d; kk++) {  // seq    Reads Q[i*Br+ii][0:d] K[j*Bc+jj][0:d]
--						float x, y;
--						
--						x = Qi[ii * d + kk];
--						y = Kj[jj * d + kk];
--						
--						sum += x * y;
--					}
--					
--					Pij[jj] = sum;
--					max = sum > max ? sum : max;
--				}
--				
--				sum = 0;
--
--				for (int jj = 0; jj < Bc; jj++) {  // segmented reduce in intra-group
--					float *x = &Pij[jj];
--					
--					sum += *x = exp(*x - max);
--				}
--				
--				mi_new = mi[ii] > max ? mi[ii] : max;
--				eij = exp(max - mi_new);
--				eli = li[ii] * exp(mi[ii] - mi_new);
--				li_new = eli + sum * eij;
--				
--				li[ii] = li_new;
--				mi[ii] = mi_new;
--				
--				for (int kk = 0; kk < d; kk++)  // a map-like update that must be mapped on parallel dimension Bc
--					Oi[ii * d + kk] *= eli;
--				
--				for (int jj = 0; jj < Bc; jj++) {  // Sequential!!!
--					float x = eij * Pij[jj];
--
--					for (int kk = 0; kk < d; kk++) { // Parallel (mapped on Bc)
--						int ik = ii * d + kk;
--						int jk = jj * d + kk;
--						
--						Oi[ik] += x * Vj[jk];  // O[i*Br+ii][kk] += eij * Pij[jj] * V[j*Bc+jj][kk]
--					}
--				}
--				
--				for (int kk = 0; kk < d; kk++)  // mapped on Bc again
--					Oi[ii * d + kk] /= li_new;
--			}
--		}
--	}
--	
--	free(l);
--	free(m);
--	free(Pij);
--	
--	return 0;
--}
--
