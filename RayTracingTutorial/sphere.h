#pragma once

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
public:
	sphere() : center({0.0, 0.0, 0.0}), radius(0.0) {}
	sphere(point3 cen, double r) : center(cen), radius(r) {}
	
	virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const override;

private:
	point3 center;
	double radius;
};

bool sphere::hit(const ray& r, double tmin, double tmax, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    double a = r.direction().length_squared();
    double half_b = dot(r.direction(), oc);
    double c = oc.length_squared() - radius * radius;
    double discriminant = half_b * half_b - a * c;

    if (discriminant <= 0) {
        return false;
    }

    double temp = std::sqrt(discriminant);
    double root = (-half_b - temp) / a;
    if (root > tmin && root < tmax) {
        rec.t = root;
        rec.p = r.at(root);
        rec.normal = (rec.p - center) / radius;
        return true;
    }

    root = (-half_b + temp) / a;
    if (root > tmin && root < tmax) {
        rec.t = root;
        rec.p = r.at(root);
        rec.normal = (rec.p - center) / radius;
        return true;
    }

    return false;
}