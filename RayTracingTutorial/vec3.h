#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>

class vec3 { 
public:
	vec3() : arr{ 0.0, 0.0, 0.0 } {}

	vec3(double x, double y, double z) : arr{ x, y, z } {}

	double x() const { return arr[0]; }
	double y() const { return arr[1]; }
	double z() const { return arr[2]; }

	vec3 operator-() const { return vec3(-arr[0], -arr[1], -arr[2]); }

	double operator[](int i) const {
		if (i < 0 || i > 2) {
			throw std::runtime_error("index out of bound.");
		}
		return arr[i];
	}

	double& operator[](int i) {
		if (i < 0 || i > 2) {
			throw std::runtime_error("index out of bound.");
		}
		return arr[i];
	}

	vec3& operator+= (const vec3& v) {
		arr[0] += v[0];
		arr[1] += v[1];
		arr[2] += v[2];

		return *this;
	}

	vec3& operator+ (const double t) {
		arr[0] += t;
		arr[1] += t;
		arr[2] += t;

		return *this;
	}

	vec3& operator*= (const double t) {
		arr[0] *= t;
		arr[1] *= t;
		arr[2] *= t;

		return *this;
	}
	vec3& operator/= (const double t) {
		arr[0] /= t;
		arr[1] /= t;
		arr[2] /= t;

		return *this;
	}
	
	double length() const { return std::sqrt(length_squared()); }

	double length_squared() const { return arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2]; }

	inline static vec3 random() {
		return vec3(random_double(), random_double(), random_double());
	}

	inline static vec3 random(double min, double max) {
		return vec3(random_double(min, max), random_double(min, max), random_double(min, max));
	}

private:
	double arr[3];
};

using point3 = vec3;
using color = vec3;

inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
	return out << v.x() << ' ' << v.y() << ' ' << v.z();
}

inline vec3 operator+(const vec3& u, const vec3& v) {
	return vec3(u.x() + v.x(), u.y() + v.y(), u.z() + v.z());
}

inline vec3 operator-(const vec3& u, const vec3& v) {
	return vec3(u.x() - v.x(), u.y() - v.y(), u.z() - v.z());
}

inline vec3 operator*(const vec3& u, const vec3& v) {
	return vec3(u.x() * v.x(), u.y() * v.y(), u.z() * v.z());
}

inline vec3 operator*(double t, const vec3& v) {
	return vec3(t * v.x(), t * v.y(), t * v.z());
}

inline vec3 operator*(const vec3& v, double t) {
	return t * v;
}

inline vec3 operator/(vec3 v, double t) {
	return (1 / t) * v;
}

inline double dot(const vec3& u, const vec3& v) {
	return u.x() * v.x() + u.y() * v.y() + u.z() * v.z();
}

inline vec3 cross(const vec3& u, const vec3& v) {
	return vec3(u.x() * v.z() - u.z() * v.y(),
				u.z() * v.x() - u.x() * v.z(),
				u.x() * v.y() - u.y() * v.x());
}

inline vec3 unit_vector(vec3 v) {
	return v / v.length();
}


inline vec3 random_in_unit_sphere() {
	while (true) {
		vec3 rand = vec3::random(-1, 1);
		if (rand.length_squared() >= 1) {
			continue;
		}

		return rand;
	}
}