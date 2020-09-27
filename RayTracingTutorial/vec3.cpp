#include "vec3.h"

vec3::vec3() : arr{0.0, 0.0, 0.0} {}

vec3::vec3(double x, double y, double z) : arr{x, y, z} {}

double vec3::x() const {
	return arr[0];
}

double vec3::y() const {
	return arr[1];
}

double vec3::z() const {
	return arr[2];
}

vec3 vec3::operator-() const {
	return vec3(-arr[0], -arr[1], -arr[2]);
}

double vec3::operator[](int i) const {
	if (i < 0 || i > 2) {
		throw std::runtime_error("index out of bound.");
	}
	return arr[i];
}

double& vec3::operator[](int i) {
	if (i < 0 || i > 2) {
		throw std::runtime_error("index out of bound.");
	}
	return arr[i];
}

vec3& vec3::operator+=(const vec3& v) {
	arr[0] += v[0];
	arr[1] += v[1];
	arr[2] += v[2];
	
	return *this;
}

vec3& vec3::operator*=(const double t)
{
	arr[0] *= t;
	arr[1] *= t;
	arr[2] *= t;

	return *this;
}

vec3& vec3::operator/=(const double t)
{
	arr[0] /= t;
	arr[1] /= t;
	arr[2] /= t;

	return *this;
}

double vec3::length() const {
	return std::sqrt(length_squared());
}

double vec3::length_squared() const {
	return arr[0] * arr[0] + arr[1] * arr[1] + arr[2] * arr[2];
}