#include <cmath>
#include <vector>
#include <omp.h>

double add(double a, double b) {
    return a + b;
}

double power(double base, double exponent) {
    return std::pow(base, exponent);
}

double parallel_sum(int n) {
    double total = 0.0;

    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < n; i++)
    {
        total += 1;
    }
    
    return total;
}