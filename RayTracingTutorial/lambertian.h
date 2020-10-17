#pragma once

#include "vec3.h"
#include "color.h"
#include "material.h"
#include "texture.h"

class lambertian : public material {
public:
	lambertian(const color& a): albedo(std::make_shared<solid_color>(a)) {}
	lambertian(std::shared_ptr<texture> a) : albedo(a) {}

	virtual bool scatter(const ray& r_in, const hit_record& rec, color& attenuation, ray& scattered) const override {
		vec3 scattered_direction = rec.normal + random_unit_vector();
		scattered = ray(rec.p, scattered_direction, r_in.time());
		attenuation = albedo->value(rec.u, rec.v, rec.p);
		return dot(scattered_direction, rec.normal) > 0;
	}

private:
	std::shared_ptr<texture> albedo;
};