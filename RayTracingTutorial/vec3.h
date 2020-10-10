#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "utilities.h"

class vec3 { 
public:
	__device__ vec3() {}

	__host__ __device__ vec3(double x, double y, double z) : arr{ x, y, z } {}

	__host__ __device__ double x() const { return arr[0]; }
	__host__ __device__ double y() const { return arr[1]; }
	__host__ __device__ double z() const { return arr[2]; }

	__host__ __device__ vec3 operator-() const { return vec3(-arr[0], -arr[1], -arr[2]); }

	__host__ __device__ double operator[](int i) const {
		assert(i >= 0 && i <= 2);
		return arr[i];
	}

	__host__ __device__ double& operator[](int i) {
		assert(i >= 0 && i <= 2);
		return arr[i];
	}

	__host__ __device__ vec3& operator+= (const vec3& v) {
		arr[0] += v[0];
		arr[1] += v[1];
		arr[2] += v[2];

		return *this;
	}

	__host__ __device__ vec3& operator+ (const double t) {
		arr[0] += t;
		arr[1] += t;
		arr[2] += t;

		return *this;
	}

	__host__ __device__ vec3& operator*= (const double t) {
		arr[0] *= t;
		arr[1] *= t;
		arr[2] *= t;

		return *this;
	}

	__host__ __device__ vec3& operator*= (const vec3& v) {
		arr[0] *= v[0];
		arr[1] *= v[1];
		arr[2] *= v[2];

		return *this;
	}

	__host__ __device__ vec3& operator/= (const double t) {
		arr[0] /= t;
		arr[1] /= t;
		arr[2] /= t;

		return *this;
	}

	__host__ __device__ double length() const { return sqrt(length_squared()); }

	__host__ __device__ double length_squared() const { return arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2]; }

	__device__ inline static vec3 random(curandState* rand_state) {
		return vec3(random_double(rand_state), random_double(rand_state), random_double(rand_state));
	}

	__device__ inline static vec3 random(double min, double max, curandState* rand_state) {
		return vec3(random_double(min, max, rand_state), random_double(min, max, rand_state), random_double(min, max, rand_state));
	}

private:
	double arr[3];
};

using point3 = vec3;
using color = vec3;

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
	return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

__host__ __device__ inline vec3 operator*(double t, const vec3& v) {
	return vec3(t * v.x(), t * v.y(), t * v.z());
}

__host__ __device__ inline vec3 operator*(const vec3& v, double t) {
	return t * v;
}

__host__ __device__ inline vec3 operator/(vec3 v, double t) {
	return (1 / t) * v;
}

__host__ __device__ inline double dot(const vec3& u, const vec3& v) {
	return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
	return vec3(u.y() * v.z() - u.z() * v.y(),
				u.z() * v.x() - u.x() * v.z(),
				u.x() * v.y() - u.y() * v.x());
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}

__device__ vec3 reflect(const vec3 & v, const vec3& n) {
	return v - 2 * dot(v, n) * n;
}

// vec3 in and n should be normalized
__device__ vec3 refract(const vec3 & in, const vec3 & n, double etai_over_etat) {
	auto cos_theta = -dot(in, n);
	vec3 r_out_perp = etai_over_etat * (in + cos_theta * n);
	vec3 r_out_parallel = -std::sqrt(fabs(1.0 - r_out_perp.length_squared())) * n;
	return r_out_parallel + r_out_perp;
}

__device__ inline vec3 random_in_unit_sphere(curandState* rand_state) {
	while (true) {
		vec3 rand = vec3::random(-1, 1, rand_state);
		if (rand.length_squared() >= 1) {
			continue;
		}

		return rand;
	}
}

// return a random point on a unit sphere centered at (0, 0, 0) with radius 1
__device__ inline vec3 random_unit_vector(curandState* rand_state) {
	double a = random_double(0, 2*pi, rand_state);
	double z = random_double(-1, 1, rand_state);
	double r = sqrt(1 - z * z);

	return vec3(r * cos(a), r * sin(a), z);
}

__device__ inline vec3 random_in_hemisphere(const vec3& normal, curandState* rand_state) {
	vec3 in_unit_sphere = random_in_unit_sphere(rand_state);
	if (dot(in_unit_sphere, normal) > 0.0) // In the same hemisphere as the normal
		return in_unit_sphere;
	else
		return -in_unit_sphere;
}

__device__ inline vec3 random_in_unit_disk(curandState* rand_state) {
	while (true) {
		auto p = vec3(random_double(-1, 1, rand_state), random_double(-1, 1, rand_state), 0);
		if (p.length() < 1.0) {
			return p;
		}
	}
}