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

const double h_pi = 3.1415926535897932385;

inline double degrees_to_radians(double degrees) {
	return degrees * (h_pi / 180.0);
}

__device__ inline double random_double(curandState* rand_state) {
	return curand_uniform(rand_state);
}

__device__ inline double random_double(double min, double max, curandState* rand_state) {
	return (max - min) * (random_double(rand_state)) + min;
}

__host__ __device__ inline double clamp(double x, double min, double max) {
	if (x < min) {
		return min;
	}
	if (x > max) {
		return max;
	}
	return x;
}