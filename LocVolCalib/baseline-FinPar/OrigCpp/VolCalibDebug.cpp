#include <cmath>

#define WITH_FLOATS     0
#define WORKGROUP_SIZE  512 

typedef double REAL;

#include "Util.h"
#include "../includeC/ParseInput.h"

using namespace std;

#define DEBUG

// Macros for 2-dim array indexing
#define Dx(i,j)       Dx[(i)*3 + j]
#define Dy(i,j)       Dy[(i)*3 + j]
#define Dxx(i,j)      Dxx[(i)*3 + j]
#define Dyy(i,j)      Dyy[(i)*3 + j]
#define MuX(i,j)      MuX[(i)*numX  + j]
#define VarX(i,j)     VarX[(i)*numX + j]
#define MuY(i,j)      MuY[(i)*numY  + j]
#define VarY(i,j)     VarY[(i)*numY + j]
#define ResultE(i,j)  ResultE[(i)*numY + j]

#define U(i,j)        U[(i)*numX + j]
#define V(i,j)        V[(i)*numY + j]

void Perturb(double *x, unsigned n, double delta)
{
//    for (size_t i = 0; i < n; i++) {
//        x[i] = x[i] * (1.0 + delta);
//    }
        x[0] = x[0] * (1.0 + delta);
        x[n - 1] = x[n - 1] * (1.0 + delta);
}

void PrintL2(const char *str, double *x, unsigned m, unsigned n)
{
    double sum = 0.0;
    for (unsigned i = 0; i < m; i++) {
        for (unsigned j = 0; j < n; j++) {
            sum += x[i * n + j] * x[i * n + j];
        }
    }
    printf("%s L2 = %e\n", str, sqrt(sum));
}

void Print(double *x, unsigned m, unsigned n)
{
    for (unsigned i = 0; i < m; i++) {
        for (unsigned j = 0; j < n; j++) {
            printf("%.17e ", x[i * n + j]);
        }
        printf("\n");
    }
}

/***********************************/

/**
 * Initializes MuX, MuY, VarX and VarY
 */
inline
void updateParams(  const unsigned numX,   
                    const unsigned numY,   
                    const unsigned g,      
                    const REAL     alpha,  
                    const REAL     beta,   
                    const REAL     nu,
                    REAL* X, REAL* Y, REAL* Time,

                    REAL* MuX, REAL* VarX, // output
                    REAL* MuY, REAL* VarY  // output
) {
    REAL c = 0.5 * nu * nu * Time[g];
#ifdef DEBUG
    printf("t = %.17e\n", Time[g]);
    printf("Y[2] = exp(%.17e)\n", Y[2] - c);
    printf("VarX[2, 0] = %.17e * %.17e = %.17e\n", pow(X[0], 2.0 * beta), 
                exp(fma(-0.5 * nu * nu, Time[g], Y[2])),
                pow(X[0], 2.0 * beta) * exp(fma(-0.5 * nu * nu, Time[g], Y[2])));
    printf("Long double Y[2] = %.17Le\n", expl((long double)Y[2] - 0.5 * nu * nu * Time[g]));
#endif
    for(unsigned j=0; j<numY; ++j) 
        for(unsigned i=0; i<numX; ++i) {
           	//MuX(j,i)  = 0.0;
            MuX(j,i)  = ((double)0.0000001) / ((numX + i) * (numY + j));    // (***Fix***)
//            VarX(j,i) = exp(2*(beta*log(X[i]) + Y[j] - 0.5*nu*nu*Time[g]));
            /* Same expression, but log(x) is ill-conditioned around 1 */
//            VarX(j, i) = pow(X[i], 2.0 * beta) * 
//                            exp(fma(-0.5 * nu * nu, Time[g], Y[j])); // (***Fix***)
            VarX(j, i) = exp(Y[j] - c) * pow(X[i], 2.0 * beta);

        }

    for(unsigned i=0; i<numX; ++i)
        for(unsigned j=0; j<numY; ++j) {
            MuY(i,j)  = 0.0;
            VarY(i,j) = nu*nu;
            //MuY(i,j) = alpha / (i * numY + j + 1);       // (***Fix***)
            //VarY(i,j) = (nu * nu) / (i * numY + j + 1);  // (***Fix***)

        }
}

/**
 * Initializes indX, indY and X, Y, Time
 */
void initGrid(  const unsigned numX, 
                const unsigned numY, 
                const unsigned numT,
                const REAL     s0, 
                const REAL     alpha, 
                const REAL     nu,
                const REAL     t,
                unsigned&      indX, // output
                unsigned&      indY, // output
                REAL*          X,    // output
                REAL*          Y,    // output
                REAL*          Time  // output
 ) {

    for(unsigned i=0; i<numT; ++i)
        Time[i] = t*i/(numT - 1);

    const REAL stdX = 20*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    indX = static_cast<unsigned>(s0/dx);
#ifdef DEBUG
    printf("indX = %d\n", indX);
    printf("dx = %.17e\n", dx);
    printf("stdX = %.17e\n", stdX);
#endif

    for(unsigned i=0; i<numX; ++i) {
        REAL ii = (REAL) i;
        X[i] = ii*log(ii+1) - indX*dx + s0;
        /* f(x) = log(x), f'(x) = 1 / |x|, so ill-conditioned around ii
         * near zero. For ii = 0, the term cancels out, so should be ok. */
        //X[i] = i*dx - indX*dx + s0;
    }

    const REAL stdY = 10.0 * nu * sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    indY = static_cast<unsigned>(numY/2);
    REAL y_constant = - (REAL)indY*dy + logAlpha;
#ifdef DEBUG
  printf("log(alpha) = %.17e\n", logAlpha);
  printf("indY = %d\n", indY);
  printf("dy = %.17e\n", dy);
  printf("y_constant = %.17e\n", y_constant);
#endif

    for(unsigned i=0; i<numY; ++i) {
        REAL ii = (REAL) i;
//        Y[i] = 0.001 * i;
        Y[i] = ii*log(ii+1)*dy + y_constant;  // (***Fix***)
//        Y[i] = ii*log(ii+1)*dy - indY*dy + logAlpha;
        //Y[i] = i*dy - indY*dy + logAlpha;
    }
}

/**
 * Initializes Globals: 
 *      (i) Dx and Dxx when called with numX and X
 *     (ii) Dy and Dyy when called with numY and Y
 */
void initOperator(  const int   n,
                    const REAL* xx, 
            
                    REAL* D,  // Output
                    REAL* DD  // Output
) {
    REAL dxl, dxu;

    //	lower boundary
    dxl		 =  0.0;
    dxu		 =  xx[1] - xx[0];

    D[0*3 + 0]  =  0.0;
    D[0*3 + 1]  = -1.0/dxu;
    D[0*3 + 2]  =  1.0/dxu;
	
    DD[0*3 + 0] =  0.0;
    DD[0*3 + 1] =  0.0;
    DD[0*3 + 2] =  0.0;
	
    //	standard case
    for(int i=1; i<n-1; i++) {
        dxl      = xx[i]   - xx[i-1];
        dxu      = xx[i+1] - xx[i];

        /* (***fix***) (same expressions, but simplified).
         * Note that D[i, 0] + D[i, 2] = 
         * (dxl^2 - dxu^2) / (dxl * dxu * (xx[i + 1] - xx[i - 1])) =
         * (dxl^2 - dxu^2) / (dxl * dxu * (dxu - dxl)) =
         * -(dxl + dxu) / (dxl * dxu)
         */
        D[i*3 + 0]  = -dxu / (dxl * (xx[i + 1] - xx[i - 1]));
        D[i*3 + 1]  = (dxu - dxl) / (dxu * dxl);
        D[i*3 + 2]  =  dxl / (dxu * (xx[i + 1] - xx[i - 1]));

        //D[i*3 + 0]  = -dxu/dxl/(dxl+dxu);
        //D[i*3 + 1]  = (dxu/dxl - dxl/dxu)/(dxl+dxu);
        //D[i*3 + 2]  =  dxl/dxu/(dxl+dxu);

        /* (***fix***) (same expressions, but simplified) */
        DD[i*3 + 0] =  2.0 / (dxl * (xx[i + 1] - xx[i - 1]));
        DD[i*3 + 1] = -2.0 / (dxu * dxl);
        DD[i*3 + 2] =  2.0 / (dxu * (xx[i + 1] - xx[i - 1]));

        //DD[i*3 + 0] =  2.0/dxl/(dxl+dxu);
        //DD[i*3 + 1] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
        //DD[i*3 + 2] =  2.0/dxu/(dxl+dxu); 
    }

    //	upper boundary
    dxl =  xx[n-1] - xx[n-2];
    dxu	=  0.0;

    D[(n-1)*3 + 0]  = -1.0/dxl;
    D[(n-1)*3 + 1]  =  1.0/dxl;
    D[(n-1)*3 + 2]  =  0.0;

    DD[(n-1)*3 + 0] = 0.0;
    DD[(n-1)*3 + 1] = 0.0;
    DD[(n-1)*3 + 2] = 0.0;
}

void setPayoff( const unsigned numX, 
                const unsigned numY, 
                const REAL     strike, 
                REAL*          X,

                REAL* ResultE  // Output

) {
    for( unsigned i=0; i<numX; ++i ) {
        REAL payoff = max( X[i]-strike, 0.0 );

        for( unsigned j=0; j<numY; ++j )
            ResultE(i,j) = payoff;
    }
}

/**
 * Computes the solution of a tridiagonal system in 
 *      output array y.
 */
inline
void tridag(    const int     n,  // input RO
                const REAL*   a,  // input RO
                      REAL*   b,  // input RW
                const REAL*   c,  // input RO
                      REAL*   y   // input & OUTPUT RW
) {
    REAL   beta;
    int    i;

    // forward swap
    for (i=1; i<n-1; i++) {
        beta = a[i] / b[i-1];

        b[i] = b[i] - beta*c[i-1];
        y[i] = y[i] - beta*y[i-1];
    }    

    // backward swap
    y[n-1] = y[n-1]/b[n-1];
    for(i=n-2; i>=0; i--) {
        y[i] = (y[i] - c[i]*y[i+1]) / b[i]; 
    }
}

void
rollback(   const unsigned numX, 
            const unsigned numY, 
            const unsigned g,
            REAL* a, REAL* b,  REAL* c, REAL* Time, REAL* U, REAL* V,
            REAL* Dx, REAL* Dxx, REAL* MuX,  REAL* VarX,
            REAL* Dy, REAL* Dyy, REAL* MuY,  REAL* VarY,

            REAL* ResultE  // output
            , unsigned numT
) {
    const REAL dtInv = 1.0 / (Time[g+1]-Time[g]);
    int        i, j;

//    PrintL2("MuX", MuX, numY, numX);
//    PrintL2("MuY", MuY, numX, numY);
//    PrintL2("VarY", VarY, numX, numY);
//    PrintL2("ResultE", ResultE, numX, numY);

//    Print(MuX, numY, numX);
//    Print(ResultE, numX, numY);
//    Print(Dxx, numX, 3);

//    Print(ResultE, numX, numY);
//    Print(VarX, numY, numX);

#ifdef DEBUG
    if (g == numT - 2) {
      printf("dtInv = %e\n", dtInv);
      printf("ResultE\n");
      Print(ResultE, numX, numY);
      printf("VarX\n");
      Print(VarX, numY, numX);
      printf("MuX\n");
      Print(MuX, numY, numX);
      printf("Dx\n");
      Print(Dx, numX, 3);
      printf("Dxx\n");
      Print(Dxx, numX, 3);
    }
#endif

    //	explicit x
    for(j=0; j<numY; j++) {
        for(i=0; i<numX; i++) {

            U(j,i) = dtInv * ResultE(i,j);

            if (0 < i) 
            U(j,i) += 0.5 * ResultE(i-1,j) * ( MuX(j,i)*Dx(i,0) + 0.5*VarX(j,i)*Dxx(i,0) );

            U(j,i) += 0.5 * ResultE(i,  j) * ( MuX(j,i)*Dx(i,1) + 0.5*VarX(j,i)*Dxx(i,1) );

            if (i < numX-1) 
            U(j,i) += 0.5 * ResultE(i+1,j) * ( MuX(j,i)*Dx(i,2) + 0.5*VarX(j,i)*Dxx(i,2) );
        }
    }

#ifdef DEBUG
    if (g == numT - 2) {
      printf("U\n");
      Print(U, numY, numX);
    }
#endif
//    Print(U, numY, numX);

//    printf("dtInv = %e\n", dtInv);
//    PrintL2("U", U, numY, numX);

	//	explicit y
    for( i=0; i<numX; i++) {
        for(j=0; j<numY; j++) {
        
            V(i,j) = 0.0;
            if (0 < j)
            V(i,j) += ResultE(i,j-1) * ( MuY(i,j)*Dy(j,0) + 0.5*VarY(i,j)*Dyy(j,0) );
            V(i,j) += ResultE(i,j  ) * ( MuY(i,j)*Dy(j,1) + 0.5*VarY(i,j)*Dyy(j,1) );
            if (j < numY-1)
            V(i,j) += ResultE(i,j+1) * ( MuY(i,j)*Dy(j,2) + 0.5*VarY(i,j)*Dyy(j,2) );

            U(j,i) += V(i,j); 
        }
    }
#ifdef DEBUG
    if (g == numT - 2) {
      printf("V\n");
      Print(V, numX, numY);
    }
#endif

//    PrintL2("V", V, numY, numX);


    //	implicit x
    for(j=0; j<numY; j++) {

        for(i=0; i<numX; i++) {
            a[i] =	     - 0.5*( MuX(j,i)*Dx(i,0) + 0.5*VarX(j,i)*Dxx(i,0) );
            b[i] = dtInv - 0.5*( MuX(j,i)*Dx(i,1) + 0.5*VarX(j,i)*Dxx(i,1) );
            c[i] =	     - 0.5*( MuX(j,i)*Dx(i,2) + 0.5*VarX(j,i)*Dxx(i,2) );
        }

        REAL* uu = U+j*numX;
        tridag(numX, a, b, c, uu);
    }

    //	implicit y
    for(i=0;i<numX;i++) {
        
        for(j=0;j<numY;j++) {
            a[j] =		 - 0.5*( MuY(i,j)*Dy(j,0) + 0.5*VarY(i,j)*Dyy(j,0) );
            b[j] = dtInv - 0.5*( MuY(i,j)*Dy(j,1) + 0.5*VarY(i,j)*Dyy(j,1) );
            c[j] =		 - 0.5*( MuY(i,j)*Dy(j,2) + 0.5*VarY(i,j)*Dyy(j,2) );
        }

        REAL* yy = ResultE + i*numY;
        for(j=0; j<numY; j++)
            yy[j] = dtInv*U(j,i) - 0.5*V(i,j);

        tridag(numY, a, b, c, yy);
    }

#ifdef DEBUG
    if (g == numT - 2) {
      printf("ResultE\n");
      Print(ResultE, numX, numY);
    }
#endif
}

REAL value(   const REAL s0,
              const REAL strike, 
              const REAL t, 
              const REAL alpha, 
              const REAL nu, 
              const REAL beta,
              const unsigned int numX,
              const unsigned int numY,
              const unsigned int numT,
              REAL* a, REAL* b,  REAL* c,   REAL* Time, REAL* U, REAL* V,
              REAL* X, REAL* Dx, REAL* Dxx, REAL* MuX,  REAL* VarX,
              REAL* Y, REAL* Dy, REAL* Dyy, REAL* MuY,  REAL* VarY,

              REAL* ResultE  // output

) {	

    unsigned indX, indY;

    initGrid    ( numX, numY, numT, 
                  s0, alpha, nu, t, 
                  indX, indY, X, Y, Time );

//    for (int i = 0; i < numX; i++) {
//        printf("%e ", X[i]);
//    }
//    printf("\n");


//    PrintL2("MuX", MuX, numY, numX);
//    PrintL2("MuY", MuY, numX, numY);
//    PrintL2("VarY", VarY, numX, numY);

    initOperator( numX, X, Dx, Dxx );
    initOperator( numY, Y, Dy, Dyy );
#ifdef DEBUG
    printf("X\n");
    Print(X, numX, 1);
    printf("Y\n");
    Print(Y, numY, 1);
#endif

//    for (int i = 0; i < 3 * numX; i += 3) {
//        printf("%e %e %e\n", Dxx[i], Dxx[i + 1], Dxx[i + 2]);
//    }
//    printf("\n");

//    PrintL2("Dx", Dx, numX, 3);
//    PrintL2("Dy", Dy, numY, 3);

    setPayoff(numX, numY, strike, X, ResultE);

    for( int i = numT-2; i>=0; --i ) {
        updateParams( numX, numY, i, alpha, beta, nu, 
                      X, Y, Time, MuX, VarX, MuY, VarY );

        rollback( numX, numY, i,  
                  a, b, c, Time, U, V,
                  Dx, Dxx, MuX, VarX,
                  Dy, Dyy, MuY, VarY,
                  ResultE, numT
                );
    }

    REAL res = ResultE(indX,indY);

    return res;
}


int main() {
    unsigned OUTER_LOOP_COUNT, numX, numY, numT; 
    REAL  s0, t, alpha, nu, beta, strike;
    REAL *a, *b, *c, *U, *V, *Time, 
         *X, *Dx, *Dxx, *MuX, *VarX,
         *Y, *Dy, *Dyy, *MuY, *VarY,
         *ResultE;

    fprintf(stdout, "\n// Original (Sequential) Volatility Calibration Benchmark:\n");
    readDataSet( OUTER_LOOP_COUNT, numX, numY, numT, s0, t, alpha, nu, beta ); 

    REAL* result = new REAL[OUTER_LOOP_COUNT];

    unsigned long int elapsed = 0;
    {   // Main Computational Kernel
        struct timeval t_start, t_end, t_diff;
        
        {  // Global Array Allocation
            const unsigned numZ = max( numX, numY );
            a       = new REAL[numZ];      // [max(numX,numY)]
            b       = new REAL[numZ];      // [max(numX,numY)]
            c       = new REAL[numZ];      // [max(numX,numY)]
            V       = new REAL[numX*numY]; // [numX, numY]
            U       = new REAL[numY*numX]; // [numY, numX]

            X       = new REAL[numX];      // [numX]
            Dx      = new REAL[numX*3];    // [numX, 3]
            Dxx     = new REAL[numX*3];    // [numX, 3]
            Y       = new REAL[numY];      // [numY]
            Dy      = new REAL[numY*3];    // [numY, 3]
            Dyy     = new REAL[numY*3];    // [numY, 3]
            Time    = new REAL[numT];      // [numT]

            MuX     = new REAL[numY*numX]; // [numY, numX]
            MuY     = new REAL[numX*numY]; // [numX, numY]
            VarX    = new REAL[numY*numX]; // [numY, numX]
            VarY    = new REAL[numX*numY]; // [numX, numY]
            ResultE = new REAL[numX*numY]; // [numX, numY]
        }

        { // Computation Kernel
            gettimeofday(&t_start, NULL);

            REAL strike;
#ifdef DEBUG
            for (unsigned i = 0; i < 1; i++) {
#else
            for( unsigned i = 0; i < OUTER_LOOP_COUNT; ++ i ) {
#endif
                strike = 0.001*i;
                result[i] = value(  s0, strike, t, 
                                    alpha, nu, beta,
                                    numX, numY, numT,
                                    a, b,  c, Time,  U, V, 
                                    X, Dx, Dxx, MuX, VarX,
                                    Y, Dy, Dyy, MuY, VarY,
                                    ResultE
                                 );
            }

            gettimeofday(&t_end, NULL);
            timeval_subtract(&t_diff, &t_end, &t_start);
            elapsed = t_diff.tv_sec*1e6+t_diff.tv_usec;
        }

        { // Global Array Deallocation
            delete[] a;    delete[] b;    delete[] c;    
            delete[] V;    delete[] U;
            delete[] X;    delete[] Dx;   delete[] Dxx;
            delete[] Y;    delete[] Dy;   delete[] Dyy;
            delete[] MuX;  delete[] MuY; 
            delete[] VarX; delete[] VarY;
            delete[] Time; delete[] ResultE;
        }
    }
        for (unsigned i = 0; i < OUTER_LOOP_COUNT; i++) {
            printf("%.17e\n", result[i]);
        }

    {   // validation and writeback of the result
        bool is_valid = validate( result, OUTER_LOOP_COUNT );
        writeStatsAndResult( is_valid, result, OUTER_LOOP_COUNT, 
                             numX, numY, numT, false, 1, elapsed );        
//        writeResult( res.data(), OUTER_LOOP_COUNT );
    }

    delete[] result;
    return 0;
}

