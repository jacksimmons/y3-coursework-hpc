/*******************************************************************************
Jack Simmons - Advection program for all tasks.
To change which task this file performs, change the value for TASK at #define TASK.
(TASK == 1 => Task 1, TASK == 4 => Task 4, etc.)


Outputs: initial.dat - inital values of u(x,y) 
         final.dat   - final values of u(x,y)

         The output files have three columns: x, y, u

         Compile with: gcc -o {filename} -std=c99 {filename}.c -lm

Notes: The time step is calculated using the CFL condition

********************************************************************************/

/*********************************************************************
                     Include header files 
**********************************************************************/

#include <stdio.h>
#include <math.h>
#include <omp.h>

/*********************************************************************
                      Main function
**********************************************************************/
#define TASK 4

int main(){
  // Distance unit: m
  // Time unit: s

  /* Grid properties */
  const int NX=1000;    // Number of x points
  const int NY=1000;    // Number of y points

  const float xmin=0.0; // Minimum x value
  const float ymin=0.0; // Minimum y value
#if TASK == 1
  const float xmax=1.0; // Maximum x value
  const float ymax=1.0; // Maximum y value
#else
  const float xmax=30.0;
  const float ymax=30.0; 
#endif
  
  /* Parameters for the Gaussian initial conditions */
#if TASK == 1
  const float x0=0.1; // Centre (x)
  const float y0=0.1; // Centre (y)
  const float sigmax=0.03; // Width (x)
  const float sigmay=0.03; // Width (y)
#else
  const float x0=3.0;                    // Centre(x)
  const float y0=15.0;                   // Centre(y)
  const float sigmax=1.0;               // Width(x)
  const float sigmay=5.0;               // Width(y)
#endif
  const float sigmax2 = sigmax * sigmax; // Width(x) squared
  const float sigmay2 = sigmay * sigmay; // Width(y) squared

  /* Boundary conditions */
  const float bval_left=0.0;    // Left boundary value
  const float bval_right=0.0;   // Right boundary value
  const float bval_lower=0.0;   // Lower boundary
  const float bval_upper=0.0;   // Upper boundary
  
  /* Time stepping parameters */
  const float CFL=0.9;   // CFL number 
#if TASK == 1
  const int nsteps=1500; // Number of time steps
#else
  const int nsteps=800;
#endif

  /* Velocity */
#if TASK == 1
  const float velx=0.01;
  const float vely=0.01;
#elif TASK == 2
  const float velx=1.0;  // Velocity in x direction
  const float vely=0.0;  // Velocity in y direction
#elif TASK >= 3
  float velx[NY+2];
  const float vely=0.0;
#endif

  /* (2.3) Shear parameters */
#if TASK >= 3
  const float U_STAR = 0.2f;    // Friction velocity
  const float Z0 = 1.0f;        // Roughness length
  const float K = 0.41f;        // Von Kármán's constant
#endif

  /* Arrays to store variables. These have NX+2 elements
     to allow boundary values to be stored at both ends */
  float x[NX+2];          // x-axis values
  float y[NX+2];          // y-axis values
  float u[NX+2][NY+2];    // Array of u values
  float dudt[NX+2][NY+2]; // Rate of change of u

  float x2;   // x squared (used to calculate iniital conditions)
  float y2;   // y squared (used to calculate iniital conditions)
  
  /* Calculate distance between points */
  float dx = (xmax-xmin) / ( (float) NX);
  float dy = (ymax-ymin) / ( (float) NY);
 
  /* Calculate time step using the CFL condition */
  /* The fabs function gives the absolute value in case the velocity is -ve */
#if TASK < 3
  float dt = CFL / ( (fabs(velx) / dx) + (fabs(vely) / dy) );
#else
  float dt = INFINITY;
#endif

  /**Report information about the calculation ***/
  printf("Grid spacing dx     = %g\n", dx);
  printf("Grid spacing dy     = %g\n", dy);
  printf("CFL number          = %g\n", CFL);
  printf("No. of time steps   = %d\n", nsteps);
#if TASK < 3
  printf("Time step           = %g\n", dt);
  printf("End time            = %g\n", dt*(float) nsteps);
  printf("Distance advected x = %g\n", velx*dt*(float) nsteps);
  printf("Distance advected y = %g\n", vely*dt*(float) nsteps);
#endif

  /*** Place x points in the middle of the cell ***/
  /* LOOP 1 */

  // Shared: x -> Each thread accesses a different index, so
  // there is no chance for race conditions. We require the full
  // array with all values entered, so shared is necessary.
  //
  // dx -> Threads require the initialised value of dx.
  #pragma omp parallel for shared(x, dx)
  for (int i=0; i<NX+2; i++){
    x[i] = ( (float) i - 0.5) * dx;
  }

  /*** Place y points in the middle of the cell ***/
  /* LOOP 2 */
  
  // Setup velocity array (if Task 3)
  // Same reasons for Loop 1.
#if TASK < 3
  #pragma omp parallel for shared(y, dy)
  for (int j=0; j<NY+2; j++){
    y[j] = ( (float) j - 0.5) * dy;
  }
#else
  #pragma omp parallel for shared(y, dy, velx, dt)
  for (int j=0; j<NY+2; j++){
	y[j] = ( (float) j - 0.5) * dy;
	if (y[j] > Z0)
		velx[j] = U_STAR * log(y[j] / Z0) / K;
	else
		velx[j] = 0;
	float thisTimeStep = CFL / ( (fabs(velx[j]) / dx) + (fabs(vely) / dy) );
	
	// Pick the MINIMUM value of dt.
	if (thisTimeStep < dt) dt = thisTimeStep;
  }
  printf("Time step           = %g\n", dt);
  printf("End time            = %g\n", dt*(float) nsteps);
#endif
  

  /*** Set up Gaussian initial conditions ***/
  /* LOOP 3 */

  // 
  // Shared: u -> Output must take writes from all threads, so must be shared.
  // Each thread has different indices, so no race conditions.
  // x, y -> Initialised outside the loop.
  // 
  // Private: x2, y2 -> Threads are reading/writing to these simultaneously,
  // so private access is necessary to avoid race conditions. Also the init'd
  // value for both is not required.
  #pragma omp parallel for shared(u) private(x2, y2) shared(x, y) collapse(2)
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      x2      = (x[i]-x0) * (x[i]-x0);
      y2      = (y[j]-y0) * (y[j]-y0);
      u[i][j] = exp( -1.0 * ( (x2/(2.0*sigmax2)) + (y2/(2.0*sigmay2)) ) );
    }
  }

  /*** Write array of initial u values out to file ***/
  FILE *initialfile;
  initialfile = fopen("initial.dat", "w");
  
  /* LOOP 4 */

  // Cannot be parallelised (without rewriting the inner code).
  //
  // The fprintf call appends a line to the file stream, which is
  // inherently a serial operation.
  // If this loop were to be parallelised, the lines would have a non-
  // deterministic order, leading to incorrect results most of the time.
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(initialfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(initialfile);
 
  /*** Update solution by looping over time steps ***/
  /* LOOP 5 */

  // It is not possible to parallelise this loop, due to the way `u` is assigned to.
  //
  // The issue is that the value of `u` at the start of every iteration depends on
  // the value at the end of the previous iteration (see loop 9), because it is a sum.
  //
  // Race conditions are also present with the use of `dudt`.
  //
  // While it may be possible to parallelise this with a rewrite of the inner loops,
  // and a `reduction` clause, parallelism cannot be attained just by adding an OpenMP
  // directive.
  for (int m=0; m<nsteps; m++){ 

    /*** Apply boundary conditions at u[0][:] and u[NX+1][:] ***/
    /* LOOP 6 */

    // Shared: u -> Threads must share this to fill the array.
    #pragma omp parallel for shared(u)
    for (int j=0; j<NY+2; j++){
      u[0][j]    = bval_left;
      u[NX+1][j] = bval_right;
    }

    /*** Apply boundary conditions at u[:][0] and u[:][NY+1] ***/
    /* LOOP 7 */

    // Same reasons as for Loop 6
    #pragma omp parallel for shared(u)
    for (int i=0; i<NX+2; i++){
      u[i][0]    = bval_lower;
      u[i][NY+1] = bval_upper;
    }
    
    /*** Calculate rate of change of u using leftward difference ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 8 (For tasks 1/2)*/
  
    // Shared: dudt -> Filled array req'd in Loop 9, each thread writes to a different index.
    // dx, dy, u -> These are only read from.
	// dt -> This is shared.
#if TASK < 3
    #pragma omp parallel for collapse(2) shared(u, dudt, dx, dy)
    for (int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
         dudt[i][j] = -velx * (u[i][j] - u[i-1][j]) / dx
	                  -vely * (u[i][j] - u[i][j-1]) / dy;
		 
       }
    }
#else
    // velx -> Only read from.
    #pragma omp parallel for collapse(2) shared(u, dudt, dx, dy, velx)
    for (int i=1; i<NX+1; i++)
    {
	  for(int j=1; j<NY+1; j++)
	  {
	    dudt[i][j] = -velx[j] * (u[i][j] - u[i-1][j]) / dx
					 -vely * (u[i][j] - u[i][j-1]) / dy;
	  }
    }
#endif
    
    /*** Update u from t to t+dt ***/
    /* Loop over points in the domain but not boundary values */
    /* LOOP 9 */
    
    // Shared: u -> Filled array req'd in next iter, each thread writes to different index.
    // dudt, dt -> Only read from.
// #if TASK < 3
    #pragma omp parallel for collapse(2) shared(u, dudt, dt)
    for	(int i=1; i<NX+1; i++){
      for (int j=1; j<NY+1; j++){
        u[i][j] = u[i][j] + dudt[i][j] * dt;
      }
    }
// #else
	// #pragma omp parallel for collapse(2) shared(u, dudt)
    // for	(int i=1; i<nx+1; i++){
      // for (int j=1; j<ny+1; j++){
		// float chosen_velx = 0.0f;
		// if (y[j] > z0)
			// chosen_velx = u_star * log(y[j] / z0) / k;
		  
		// if (y[j] > z0)
			// chosen_dt = cfl / ( (fabs(chosen_velx) / dx) + (fabs(vely) / dy) );
        // u[i][j] = u[i][j] + dudt[i][j] * chosen_dt;
      // }
    // }
// #endif
  } // time loop
 
 
  /*** Write array of final u values out to file ***/
  FILE *finalfile;
  finalfile = fopen("final.dat", "w");
  
  /* LOOP 10 */

  // Cannot be parallelised. (Same reason as Loop 4)
  for (int i=0; i<NX+2; i++){
    for (int j=0; j<NY+2; j++){
      fprintf(finalfile, "%g %g %g\n", x[i], y[j], u[i][j]);
    }
  }
  fclose(finalfile);
  
  // Vertically averaged plot (excluding BCs)
  // A plot of x against vertical average
#if TASK == 4
  FILE *vAvgPlotFile;
  vAvgPlotFile = fopen("vavgplot.dat", "w");
  for (int i=1; i<NX+1; i++){
	float sum = 0;
	for (int j=1; j<NY+1; j++){
	  sum += u[i][j];
	}
	fprintf(vAvgPlotFile, "%g %g\n", x[i], sum/NY);
  }
  fclose(vAvgPlotFile);
#endif

  return 0;
}

/* End of file ******************************************************/
