# run with
# python3 c_from_python.py

import ctypes
import pathlib
# import faulthandler
# faulthandler.enable()

# tell python the library is in the current directory
libname = pathlib.Path().absolute() / "chipot_cpp_wrapper.so"
print("libname:", libname)

# load library
my_functions = ctypes.CDLL(libname)
# tell python what arguments the function needs
my_functions.chipot_real_wrapper.restype = ctypes.c_double
my_functions.chipot_real_wrapper.argtypes = [ ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

my_functions.chipot_imag_wrapper.restype = ctypes.c_double
my_functions.chipot_imag_wrapper.argtypes = [ ctypes.c_int, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]

x = my_functions.chipot_regression_test()
print("x is:",x)

ps =  ctypes.c_int(1)
pt =  ctypes.c_int(1)
ppx = ctypes.c_int(0)
ppy = ctypes.c_int(0)
ppz = ctypes.c_int(0)

qs =  ctypes.c_int(-1)
qt =  ctypes.c_int(1)
qpx = ctypes.c_int(1)
qpy = ctypes.c_int(0)
qpz = ctypes.c_int(0)

rs =  ctypes.c_int(1)
rt =  ctypes.c_int(1)
rpx = ctypes.c_int(1)
rpy = ctypes.c_int(0)
rpz = ctypes.c_int(0)

ss =  ctypes.c_int(-1)
st =  ctypes.c_int(1)
spx = ctypes.c_int(0)
spy = ctypes.c_int(0)
spz = ctypes.c_int(0)

nParticles = ctypes.c_int(4)
rho = ctypes.c_double(0.08)

mat_elem_real = ctypes.c_double(1.0)
mat_elem_imag = ctypes.c_double(-1.0)

print("real:", mat_elem_real, "imag:", mat_elem_imag)
print("real:", mat_elem_real.value, "imag:", mat_elem_imag.value)

mat_elem_real = my_functions.chipot_real_wrapper(
        nParticles, rho,
        ps, pt, ppx, ppy, ppz,
        qs, qt, qpx, qpy, qpz,
        rs, rt, rpx, rpy, rpz,
        ss, st, spx, spy, spz)

mat_elem_imag = my_functions.chipot_imag_wrapper(
        nParticles, rho,
        ps, pt, ppx, ppy, ppz,
        qs, qt, qpx, qpy, qpz,
        rs, rt, rpx, rpy, rpz,
        ss, st, spx, spy, spz)

print("real:", mat_elem_real, "imag:", mat_elem_imag)

