#ifndef chipot_cpp_wrapper_hpp_
#define chipot_cpp_wrapper_hpp_
#include <complex>
#include "SPBasis.hpp"

// this is the format that my cpp code prefers to call
std::complex<double> chipot_cpp_wrapper(SPBasis *basis, double density, int p, int q, int r, int s);

// wrapper for the fortran code
extern "C"
void chipot_f90_wrapper(double *matel_real, double *matel_im,
							int Nparticles, double rho,
							int ps, int pt, int ppx, int ppy, int ppz,
							int qs, int qt, int qpx, int qpy, int qpz,
							int rs, int rt, int rpx, int rpy, int rpz,
							int ss, int st, int spx, int spy, int spz);

extern "C"
double chipot_real_wrapper(int Nparticles, double rho,
							int ps, int pt, int ppx, int ppy, int ppz,
							int qs, int qt, int qpx, int qpy, int qpz,
							int rs, int rt, int rpx, int rpy, int rpz,
							int ss, int st, int spx, int spy, int spz);

extern "C"
double chipot_imag_wrapper(int Nparticles, double rho,
							int ps, int pt, int ppx, int ppy, int ppz,
							int qs, int qt, int qpx, int qpy, int qpz,
							int rs, int rt, int rpx, int rpy, int rpz,
							int ss, int st, int spx, int spy, int spz);


// check that at least one matrix element matches up
extern "C"
int chipot_regression_test();

#endif
