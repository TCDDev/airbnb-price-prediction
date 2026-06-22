#pragma once

namespace geo {

    constexpr double EARTH_RADIUS_KM = 6371.0088;

    double fast_haversine(
        double lat1_deg,
        double lon1_deg,
        double lat2_deg,
        double lon2_deg

    );

}