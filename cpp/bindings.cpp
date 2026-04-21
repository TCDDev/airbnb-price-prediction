// NOTE:
// VSCode may show an include error for <pybind11/pybind11.h>.
// This is an IntelliSense issue. The project builds correctly via CMake/scikit-build.
// Safe to ignore unless the actual build fails.

#include <pybind11/pybind11.h>

namespace py = pybind11;

double add(double a, double b);
double power(double base, double exponent);
double parallel_sum(int n);

PYBIND11_MODULE(fast_module, m) {
    m.def("add", &add, "Add two numbers");
    m.def("power", &power, "Raise a number to a power");
    m.def("parallel_sum", &parallel_sum, "Parallelized summing with OpenMP");
}