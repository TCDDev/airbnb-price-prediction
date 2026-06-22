#pragma once

#include <vector>

namespace geo {

    constexpr double EARTH_RADIUS_KM = 6371.0088;

    double fast_haversine(
        double lat1_deg,
        double lon1_deg,
        double lat2_deg,
        double lon2_deg

    );

    std::vector<double> fast_haversine_batch (
        const std::vector<double>& lat1,
        const std::vector<double>& lat2,
        const std::vector<double>& lon1,
        const std::vector<double>& lon2
    );

}