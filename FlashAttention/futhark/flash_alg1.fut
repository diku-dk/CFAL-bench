------------------------
--- In-Progress Work ---
------------------------

-- N = m*d   4*d | M?

def FlashAttention [d][m]
        (M : i64)
        (O: [m*d][d]real)
        (Q: [m*d][d]real) 
        (K: [m*d][d]real) 
        (V: [m*d][d]real) 
      : [m][d][d]real =
  
  let Bc = M / (4 * d)
  let Br = i64.min d Bc -- d < Bc ? d : Bc;
  let Tr = N / Br
  let Tc = N / Bc
  
  let res =
    tabulate_2d Tc Tr
      (\ j i ->
        let Kj = K[j*Bc : j*(Bc+1)]
		let Vj = V[j*Bc : j*(Bc+1)]
        let Oi = O[i*Br : i*(Br+1)]
        let Qi = Q[i*Br : i*(Br+1)]
        in
        tabulate Br 
          (\ ii -> 
            
          )
        

        
        
-----------------------------
--- The C implementation: ---
-----------------------------

int
flash_attention(float *O, float *Q, float *K, float *V, int N, int d, int M)
{
	float *l, *m, *Pij;
	int Br, Bc, Tr, Tc;
	
	Bc = M / (4 * d);
	Br = d < Bc ? d : Bc;   // Br = min (d, Bc)
	Tr = N / Br;
	Tc = N / Bc;
	
	if ((l = calloc(N, sizeof(float))) == NULL)
		return 1;
	
	if ((m = calloc(N, sizeof(float))) == NULL)
		return 1;
	
	if ((Pij = calloc(Bc, sizeof(float))) == NULL)
		return 1;
	
	for (int i = 0; i < N; i++)
		m[i] = -INFINITY;
	
	for (int j = 0; j < Tc; j++) {
		float *Kj, *Vj;
		
		Kj = K + j * Bc * d;
		Vj = V + j * Bc * d;
		
		for (int i = 0; i < Tr; i++) {
			float *Oi, *Qi, *mi, *li;
			
			Oi = O + i * Br * d;
			Qi = Q + i * Br * d;
			mi = m + i * Br;
			li = l + i * Br;
			
			// READS:     Q[i*Br : i*(Br+1)]   K[j*Bc: j*(Bc+1)]
			// COMPUTES:  Pij[0:Br][0:Bc]
			for (int ii = 0; ii < Br; ii++) {  // par?
				float sum, max, li_new, mi_new, eij, eli;
				
				max = -INFINITY;

				for (int jj = 0; jj < Bc; jj++) { // par?
					sum = 0;
					
					for (int kk = 0; kk < d; kk++) {  // seq    Reads Q[i*Br+ii][0:d] K[j*Bc+jj][0:d]
						float x, y;
						
						x = Qi[ii * d + kk];
						y = Kj[jj * d + kk];
						
						sum += x * y;
					}
					
					Pij[jj] = sum;
					max = sum > max ? sum : max;
				}
				
				sum = 0;

				for (int jj = 0; jj < Bc; jj++) {  // segmented reduce in intra-group
					float *x = &Pij[jj];
					
					sum += *x = exp(*x - max);
				}
				
				mi_new = mi[ii] > max ? mi[ii] : max;
				eij = exp(max - mi_new);
				eli = li[ii] * exp(mi[ii] - mi_new);
				li_new = eli + sum * eij;
				
				li[ii] = li_new;
				mi[ii] = mi_new;
				
				for (int kk = 0; kk < d; kk++)  // a map-like update that must be mapped on parallel dimension Bc
					Oi[ii * d + kk] *= eli;
				
				for (int jj = 0; jj < Bc; jj++) {  // Sequential!!!
					float x = eij * Pij[jj];

					for (int kk = 0; kk < d; kk++) { // Parallel (mapped on Bc)
						int ik = ii * d + kk;
						int jk = jj * d + kk;
						
						Oi[ik] += x * Vj[jk];  // O[i*Br+ii][kk] += eij * Pij[jj] * V[j*Bc+jj][kk]
					}
				}
				
				for (int kk = 0; kk < d; kk++)  // mapped on Bc again
					Oi[ii * d + kk] /= li_new;
			}
		}
	}
	
	free(l);
	free(m);
	free(Pij);
	
	return 0;
}

