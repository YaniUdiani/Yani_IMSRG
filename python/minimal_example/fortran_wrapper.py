# run with
# python3 c_from_python.py

import ctypes
import pathlib

# tell python the library is in the current directory
libname = pathlib.Path().absolute() / "testlib.so"
print("libname:", libname)

# load library
my_functions = ctypes.CDLL(libname)
# tell python what arguments the function needs
# my_functions.timestwo.argtypes = [ctypes.c_int]

# 8 times 2 better be 16!
x = 8
y = my_functions.timestwo(x)
print(x,"times two is:",y)

y = my_functions.timesthree_fortran(x)
print(x,"times three is:",y)
