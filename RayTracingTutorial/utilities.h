#pragma once

#include<curand.h>
#include<curand_kernel.h>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

// constants

__constant__ double infinity = std::numeric_limits<double>::infinity();
__constant__ double pi = 3.1415926535897932385;

__device__ inline double degrees_to_radians(double degrees) {
	return degrees * (pi / 180.0);
}

__device__ inline double random_double() {
	return (rand() * 1.0) / (1.0 + RAND_MAX);
}

__device__ inline double random_double(double min, double max) {
	return (max - min) * (random_double()) + min;
}

__device__ inline double clamp(double x, double min, double max) {
	if (x < min) {
		return min;
	}
	if (x > max) {
		return max;
	}
	return x;
}