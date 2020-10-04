#pragma once

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <limits>
#include <memory>

// constants

const double infinity = std::numeric_limits<double>::infinity();
const double pi = 3.1415926535897932385;

inline double degrees_to_radians(double degrees) {
	return degrees * (pi / 180.0);
}

inline double random_double() {
	return (rand() * 1.0) / (1.0 + RAND_MAX);
}

inline double random_double(double min, double max) {
	return (max - min) * (random_double()) + min;
}

inline double clamp(double x, double min, double max) {
	if (x < min) {
		return min;
	}
	if (x > max) {
		return max;
	}
	return x;
}