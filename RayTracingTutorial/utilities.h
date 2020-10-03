#pragma once

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>
#include <vector>
#include <stdexcept>

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

inline std::vector<int> find_closest_factors(int num) {
	if (num <= 0) {
		throw std::runtime_error("num should be positve.");
	}

	for (int i = (int) std::sqrt(num); i >= 1; --i) {
		if (num % i == 0) {
			return { num / i, i };
		}
	}

	assert(false && "two positive factors should exists for postive integer num.");
	return {};
}