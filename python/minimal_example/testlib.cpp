// compile into shared library with
// g++ -fPIC -shared -o testlib.so testlib.cpp

// extern c let's the compiler know that this function maybe be called from non-c programs.
// In this case python, but also if it will be called from fortran or c++

extern "C"
int timesthree_fortran(int in);

extern "C"
int timestwo(int in){
  /* return timesthree_fortran(in); */
  return 2*in;
}

