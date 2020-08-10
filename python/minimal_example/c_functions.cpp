// compile into shared library with
// g++ -fPIC -shared -o testlib.so testlib.cpp

// extern c let's the compiler know that this function maybe be called from non-c programs.
// In this case python, but also if it will be called from fortran or c++


// c wrapper for fortran code
// note, fortran function needs a "bind(c)"
extern "C"
int timesthree_fortran(int in);

// c code callable from python
extern "C"
int timestwo(int in){
  return 2*in;
}

