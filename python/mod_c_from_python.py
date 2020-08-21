import ctypes
import pathlib

# tell python the library is in the current directory
libname = pathlib.Path().absolute() / "chipot_cpp_wrapper.so"
print("lib name containing .so file:", libname)

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



