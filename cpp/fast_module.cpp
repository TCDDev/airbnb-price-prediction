#include <cmath>
#include <omp.h>
#include "haversine.hpp"

// Toy functions for testing purposes

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

// native fast haversine distance

namespace geo {

    namespace {
        constexpr double PI = 3.14159265358979323846;

        double deg_to_rad(double degrees) {
            return degrees * PI / 180.00;
        }
    }

    double fast_haversine(double lat1_deg, double lon1_deg, double lat2_deg, double lon2_deg) {
        
        const double lat1 = deg_to_rad(lat1_deg);
        const double lat2 = deg_to_rad(lat2_deg);
        const double lon1 = deg_to_rad(lon1_deg);
        const double lon2 = deg_to_rad(lon2_deg);

        const double dlat = lat2 - lat1;
        const double dlon = lon2 - lon1;

        const double sin_dlat = std::sin(dlat / 2.0);
        const double sin_dlon = std::sin(dlon / 2.0);

        const double a = sin_dlat * sin_dlat + std::cos(lat1) * std::cos(lat2) * sin_dlon * sin_dlon;

        const double c = 2.0 * std::asin(std::sqrt(a));

        return EARTH_RADIUS_KM * c;

    }
}