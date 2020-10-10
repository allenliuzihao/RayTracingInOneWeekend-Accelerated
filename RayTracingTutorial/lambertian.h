#pragma once

#include "vec3.h"
#include "color.h"
#include "material.h"

class lambertian : public material {
public:
	__device__ lambertian(const color& a) : albedo(a) {}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered, curandState* rand_state) const override {
		vec3 scattered_direction = rec.normal + random_unit_vector(rand_state);
		scattered = ray(rec.p, scattered_direction);
		attenuation = albedo;
		return dot(scattered_direction, rec.normal) > 0;
	}

private:
	color albedo;
};