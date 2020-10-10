#pragma once

#include "ray.h"
#include "color.h"
#include "material.h"

class metal : public material {
public:
	__device__ metal(const color& a, double f) : albedo(a), fuzz(f < 1 ? f : 1) {}

	__device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		attenuation = albedo;
		scattered = ray(rec.p, reflected + fuzz * random_unit_vector());
		return dot(scattered.direction(), rec.normal) > 0;
	}

private:
	color albedo;
	double fuzz;
};