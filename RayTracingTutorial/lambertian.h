#pragma once

#include "vec3.h"
#include "color.h"
#include "material.h"

class lambertian : public material {
public:
	lambertian(const color& a): albedo(a) {}

	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
		vec3 scattered_direction = rec.normal + random_unit_vector();
		scattered = ray(rec.p, scattered_direction, r_in.time());
		attenuation = albedo;
		return dot(scattered_direction, rec.normal) > 0;
	}

private:
	color albedo;
};