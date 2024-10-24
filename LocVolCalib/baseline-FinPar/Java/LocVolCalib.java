/**
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
**/

class LocVolCalib {
  public static void
  updateParams( int numX,   
                int numY,   
                int g,      
                double     alpha,  
                double     beta,   
                double     nu,
                double[] X, double[] Y, double[] Time,

                double[] MuX, double[] VarX, // output
                double[] MuY, double[] VarY  // output
  ) {

    for(int j=0; j<numY; ++j) 
        for(int i=0; i<numX; ++i) {
           	//MuX(j,i)  = 0.0;
            MuX [j*numX + i] = ((double)0.0000001) / ((numX + i) * (numY + j));    // (***Fix***)
            VarX[j*numX + i] = Math.exp(2*( beta*Math.log(X[i]) + Y[j] - 0.5*nu*nu*Time[g] ));
        }

    for(int i=0; i<numX; ++i)
        for(int j=0; j<numY; ++j) {
            //MuY(i,j)  = 0.0;
            //VarY(i,j) = nu*nu;
            MuY [i*numY + j] = alpha / (i * numY + j + 1);       // (***Fix***)
            VarY[i*numY + j] = (nu * nu) / (i * numY + j + 1);  // (***Fix***)

        }
  }

  //////////////////////////////////
  
  public static int
  initGrid( int   numX, 
            int   numY, 
            int   numT,
            double  s0, 
            double  alpha, 
            double  nu,
            double  t,
            double[] X,    // output
            double[] Y,    // output
            double[] Time  // output
  ) {

    for(int i=0; i<numT; ++i)
        Time[i] = t*i/(numT-1);

    double stdX = 20.0 * alpha * s0 * Math.sqrt(t);
    double dx = stdX/numX;
    int indX = (int)(s0/dx);

    for(int i=0; i<numX; ++i) {
        double ii = (double) i;
        X[i] = ii*Math.log(ii+1)*dx - indX*dx + s0;       // (***Fix***)
        //X[i] = i*dx - indX*dx + s0;
    }

    double stdY = 10.0 * nu * Math.sqrt(t);
    double dy = stdY/numY;
    double logAlpha = Math.log(alpha);
    
    // indY = static_cast<unsigned>(numY/2);
    int indY = numY / 2;
    
    for(int i=0; i<numY; ++i) {
        double ii = (double) i;
        Y[i] = ii*Math.log(ii+1)*dy - indY*dy + logAlpha;  // (***Fix***)
        //Y[i] = i*dy - indY*dy + logAlpha;
    }
    return indX;
  }

  //////////////////////////////////////

  public static void
  initOperator( int   n,
                double[] xx, 
            
                double[] D,  // Output
                double[] DD  // Output
  ) {
    double dxl, dxu;

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

        D[i*3 + 0]  = -dxu/dxl/(dxl+dxu);
        D[i*3 + 1]  = (dxu/dxl - dxl/dxu)/(dxl+dxu);
        D[i*3 + 2]  =  dxl/dxu/(dxl+dxu);

        DD[i*3 + 0] =  2.0/dxl/(dxl+dxu);
        DD[i*3 + 1] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
        DD[i*3 + 2] =  2.0/dxu/(dxl+dxu); 
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

  ///////////////////////////////////////////
  
  public static void
  setPayoff( int   numX, 
             int   numY, 
             double  strike, 
             double[] X,

             double[] ResultE  // Output
  ) {
    for( int i=0; i<numX; ++i ) {
        double payoff = Math.max( X[i]-strike, 0.0 );

        for( int j=0; j<numY; ++j )
            ResultE[i*numY + j] = payoff;
    }
  }
  
  ///////////////////////////////////////////
 
  public static void
  tridag( int     n,  // input RO
          double[]   a,  // input RO
          double[]   b,  // input RW
          double[]   c,  // input RO
          double[]   y,  // input & OUTPUT RW
          int        offset
  ) {
    double   beta;
    int    i;

    // forward swap
    for (i=1; i<n-1; i++) {
        beta = a[i] / b[i-1];

        b[i] = b[i] - beta*c[i-1];
        y[offset + i] = y[offset + i] - beta*y[offset + i-1];
    }    

    // backward swap
    y[offset + n-1] = y[offset + n-1]/b[n-1];
    for(i=n-2; i>=0; i--) {
        y[offset + i] = (y[offset + i] - c[i]*y[offset + i+1]) / b[i]; 
    }
  }
 
  ///////////////////////////////////////////////
 
  // #define U(i,j)        U[(i)*numX + j]
  // #define V(i,j)        V[(i)*numY + j]
  // #define ResultE(i,j)  ResultE[(i)*numY + j]
  // #define MuX(i,j)      MuX[(i)*numX  + j]
  // #define MuY(i,j)      MuY[(i)*numY  + j]
  // #define VarY(i,j)     VarY[(i)*numY + j]
  // #define VarX(i,j)     VarX[(i)*numX + j]
  public static void
  rollback( int   numX, 
            int   numY, 
            int   g,
            double[] a, double[] b,  double[] c, double[] Time, double[] U, double[] V,
            double[] Dx, double[] Dxx, double[] MuX,  double[] VarX,
            double[] Dy, double[] Dyy, double[] MuY,  double[] VarY,

            double[] ResultE  // output
  ) {
    double dtInv = 1.0 / (Time[g+1]-Time[g]);
    int  i, j;

    //	explicit x
    for(j=0; j<numY; j++) {
        for(i=0; i<numX; i++) {
            double varx = VarX[j*numX + i];
            double mux  = MuX[j*numX + i];
            U[j*numX + i] = dtInv * ResultE[i*numY + j];

            if (0 < i) 
            U[j*numX + i] += 0.5 * ResultE[ (i-1)*numY + j] * ( mux*Dx[i*3 + 0] + 0.5*varx*Dxx[i*3 + 0] );

            U[j*numX + i] += 0.5 * ResultE[i*numY + j] * ( mux*Dx[i*3 + 1] + 0.5*varx*Dxx[i*3 + 1] );

            if (i < numX-1) 
            U[j*numX + i] += 0.5 * ResultE[(i+1)*numY + j] * ( mux*Dx[i*3 + 2] + 0.5*varx*Dxx[i*3 + 2] );
        }
    }

	//	explicit y
    for( i=0; i<numX; i++) {
        for(j=0; j<numY; j++) {
            double vary = VarY[i*numY + j];
            double muy  = MuY[i*numY + j];
            V[i*numY + j] = 0.0;
            if (0 < j)
            V[i*numY + j] += ResultE[i*numY + j-1] * ( muy*Dy[j*3 + 0] + 0.5*vary*Dyy[j*3 + 0] );
            V[i*numY + j] += ResultE[i*numY + j  ] * ( muy*Dy[j*3 + 1] + 0.5*vary*Dyy[j*3 + 1] );
            if (j < numY-1)
            V[i*numY + j] += ResultE[i*numY + j+1] * ( muy*Dy[j*3 + 2] + 0.5*vary*Dyy[j*3 + 2] );

            U[j*numX + i] += V[i*numY + j]; 
        }
    }

    //	implicit x
    for(j=0; j<numY; j++) {

        for(i=0; i<numX; i++) {
            double varx = VarX[j*numX + i];
            double mux  = MuX[j*numX + i];
            a[i] =	     - 0.5*( mux*Dx[i*3 + 0] + 0.5*varx*Dxx[i*3 + 0] );
            b[i] = dtInv - 0.5*( mux*Dx[i*3 + 1] + 0.5*varx*Dxx[i*3 + 1] );
            c[i] =	     - 0.5*( mux*Dx[i*3 + 2] + 0.5*varx*Dxx[i*3 + 2] );
        }

        //REAL* uu = U+j*numX;
        tridag(numX, a, b, c, U, j*numX);
    }

    //	implicit y
    for(i=0;i<numX;i++) {
        
        for(j=0;j<numY;j++) {
            double vary = VarY[i*numY + j];
            double muy  = MuY[i*numY + j];
            a[j] =		 - 0.5*( muy*Dy[j*3 + 0] + 0.5*vary*Dyy[j*3 + 0] );
            b[j] = dtInv - 0.5*( muy*Dy[j*3 + 1] + 0.5*vary*Dyy[j*3 + 1] );
            c[j] =		 - 0.5*( muy*Dy[j*3 + 2] + 0.5*vary*Dyy[j*3 + 2] );
        }

        //REAL* yy = ResultE + i*numY;
        //for(j=0; j<numY; j++)
            //yy[j] = dtInv*U[j*numX + i] - 0.5*V[i*numY + j];

        //tridag(numY, a, b, c, yy);
        for(j=0; j<numY; j++)
            ResultE[i*numY + j] = dtInv*U[j*numX + i] - 0.5*V[i*numY + j];

        tridag(numY, a, b, c, ResultE, i*numY);

    }
  }
  
  ///////////////////////////////////////////
  
  public static double
  value( double s0,
         double strike, 
         double t, 
         double alpha, 
         double nu, 
         double beta,
         int numX,
         int numY,
         int numT,
         double[] a, double[] b,  double[] c,   double[] Time, double[] U, double[] V,
         double[] X, double[] Dx, double[] Dxx, double[] MuX,  double[] VarX,
         double[] Y, double[] Dy, double[] Dyy, double[] MuY,  double[] VarY,

         double[] ResultE  // output
  ) {	

    int indY = numY / 2;

    int indX = LocVolCalib.initGrid
                ( numX, numY, numT, 
                  s0, alpha, nu, t, 
                  X, Y, Time
                );

    LocVolCalib.initOperator( numX, X, Dx, Dxx );
    LocVolCalib.initOperator( numY, Y, Dy, Dyy );

    LocVolCalib.setPayoff(numX, numY, strike, X, ResultE);

    for( int i = numT-2; i>=0; --i ) {
        LocVolCalib.updateParams
          ( numX, numY, i, alpha, beta, nu, 
            X, Y, Time, MuX, VarX, MuY, VarY
          );

        LocVolCalib.rollback
          ( numX, numY, i,  
            a, b, c, Time, U, V,
            Dx, Dxx, MuX, VarX,
            Dy, Dyy, MuY, VarY,
            ResultE
          );
    }

    double res = ResultE[indX*numY + indY];

    return res;
  }

 
  ///////////////////////////////////////////
  public static void main(String args[]){  
    int OUTER_LOOP_COUNT, numX, numY, numT; 
    double  s0, t, alpha, nu, beta;
    double[] a, b, c, U, V, Time, 
         X, Dx, Dxx, MuX, VarX,
         Y, Dy, Dyy, MuY, VarY,
         ResultE;

    System.out.println("\n// Original (Sequential) Volatility Calibration Benchmark:");
    {
      int count = 0;
      OUTER_LOOP_COUNT = Integer.parseInt(args[count++]);
      numX = Integer.parseInt(args[count++]);
      numY = Integer.parseInt(args[count++]);
      numT = Integer.parseInt(args[count++]);
      s0   = Double.parseDouble(args[count++]); 
      t    = Double.parseDouble(args[count++]);
      alpha= Double.parseDouble(args[count++]);
      nu   = Double.parseDouble(args[count++]);
      beta = Double.parseDouble(args[count++]);
    }
//    System.out.print("OUTER: ");
//    System.out.println(OUTER_LOOP_COUNT);

    double[] result = new double[OUTER_LOOP_COUNT];

    {   // Main Computational Kernel
        
        {  // Global Array Allocation
            int numZ = Math.max( numX, numY );
            a       = new double[numZ];      // [max(numX,numY)]
            b       = new double[numZ];      // [max(numX,numY)]
            c       = new double[numZ];      // [max(numX,numY)]
            V       = new double[numX*numY]; // [numX, numY]
            U       = new double[numY*numX]; // [numY, numX]

            X       = new double[numX];      // [numX]
            Dx      = new double[numX*3];    // [numX, 3]
            Dxx     = new double[numX*3];    // [numX, 3]
            Y       = new double[numY];      // [numY]
            Dy      = new double[numY*3];    // [numY, 3]
            Dyy     = new double[numY*3];    // [numY, 3]
            Time    = new double[numT];      // [numT]

            MuX     = new double[numY*numX]; // [numY, numX]
            MuY     = new double[numX*numY]; // [numX, numY]
            VarX    = new double[numY*numX]; // [numY, numX]
            VarY    = new double[numX*numY]; // [numX, numY]
            ResultE = new double[numX*numY]; // [numX, numY]
        }

        { // Computation Kernel

            double strike;
            for( int i = 0; i < OUTER_LOOP_COUNT; ++ i ) {
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

        }
    }
    
    System.out.println("Final Result");
    for(int q=0; q<OUTER_LOOP_COUNT; q++) {
        System.out.print(result[q]);
        System.out.print(", ");
    }
    System.out.println();
  }  
}  
