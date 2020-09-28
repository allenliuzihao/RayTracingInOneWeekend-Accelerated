#pragma once

#include "ray.h"
#include "color.h"
#include "material.h"

class metal : public material {
public:
	metal(const color& a) : albedo(a) {}

	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
		vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
		attenuation = albedo;
		scattered = ray(rec.p, reflected);
		return dot(reflected, rec.normal) > 0;
	}

private:
	color albedo;
};