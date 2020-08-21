from mod_c_from_python import my_functions
import ctypes

"Returns anti-symmetrized matrix elements"

def H2B_chiral_element(p, q, r, s, full_state, rho):
  
  temp_p = full_state[p]
  
  ps =  ctypes.c_int(temp_p[3])
  pt =  ctypes.c_int(-1) #PNM
  ppx = ctypes.c_int(temp_p[0])
  ppy = ctypes.c_int(temp_p[1])
  ppz = ctypes.c_int(temp_p[2])

  temp_q = full_state[q]  
  
  qs =  ctypes.c_int(temp_q[3])
  qt =  ctypes.c_int(-1)
  qpx = ctypes.c_int(temp_q[0])
  qpy = ctypes.c_int(temp_q[1])
  qpz = ctypes.c_int(temp_q[2])

  temp_r = full_state[r]
  
  rs =  ctypes.c_int(temp_r[3])
  rt =  ctypes.c_int(-1)
  rpx = ctypes.c_int(temp_r[0])
  rpy = ctypes.c_int(temp_r[1])
  rpz = ctypes.c_int(temp_r[2])
 
  temp_s = full_state[s]
  
  ss =  ctypes.c_int(temp_s[3])
  st =  ctypes.c_int(-1)
  spx = ctypes.c_int(temp_s[0])
  spy = ctypes.c_int(temp_s[1])
  spz = ctypes.c_int(temp_s[2])
  
  nParticles = ctypes.c_int(4)
  rho = ctypes.c_double(rho)
  
#  mat_elem_real = ctypes.c_double(1.0)
#  mat_elem_imag = ctypes.c_double(-1.0)
  
  #print("real:", mat_elem_real, "imag:", mat_elem_imag)
  #print("real:", mat_elem_real.value, "imag:", mat_elem_imag.value)
  
  rs_mat_elem_real = my_functions.chipot_real_wrapper(
          nParticles, rho,
          ps, pt, ppx, ppy, ppz,
          qs, qt, qpx, qpy, qpz,
          rs, rt, rpx, rpy, rpz,
          ss, st, spx, spy, spz)

  sr_mat_elem_real = my_functions.chipot_real_wrapper(#for anti-symmetrization
          nParticles, rho,
          ps, pt, ppx, ppy, ppz,
          qs, qt, qpx, qpy, qpz,
          ss, st, spx, spy, spz,
          rs, rt, rpx, rpy, rpz)  
  
  rs_mat_elem_imag = my_functions.chipot_imag_wrapper(
          nParticles, rho,
          ps, pt, ppx, ppy, ppz,
          qs, qt, qpx, qpy, qpz,
          rs, rt, rpx, rpy, rpz,
          ss, st, spx, spy, spz)

  sr_mat_elem_imag = my_functions.chipot_imag_wrapper(#for anti-symmetrization
          nParticles, rho,
          ps, pt, ppx, ppy, ppz,
          qs, qt, qpx, qpy, qpz,
          ss, st, spx, spy, spz,
          rs, rt, rpx, rpy, rpz)  
  
  #print("real:", mat_elem_real, "imag:", mat_elem_imag)
  
  return rs_mat_elem_real - sr_mat_elem_real, rs_mat_elem_imag - sr_mat_elem_imag


