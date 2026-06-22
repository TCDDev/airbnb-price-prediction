#include <cmath>
#include <cstddef>
#include <omp.h>
#include <stdexcept>
#include <vector>
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

    std::vector<double> fast_haversine_batch(
        const std::vector<double> &lat1, 
        const std::vector<double> &lat2, 
        const std::vector<double> &lon1, 
        const std::vector<double> &lon2
    ) {
        const std::size_t n = lat1.size();

        if (
            lon1.size() != n ||
            lat2.size() != n ||
            lon2.size() != n
        ) {
            throw std::invalid_argument ("All input vectors must be of the same size");
        }

        std::vector<double> distances(n);

        #pragma omp parallel for
        for (int i = 0; i < static_cast<int>(n); ++i) {
            distances[i] = fast_haversine(lat1[i], lon1[i], lat2[i], lon2[i]);
        }
    
        return distances;
    }

    
}