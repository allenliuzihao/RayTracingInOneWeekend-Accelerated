#pragma once

#include "utilities.h"

#include "hittable.h"

class moving_sphere : public hittable {
public:
	moving_sphere(): radius(0.0), time0(0.0), time1(0.0) {}

	moving_sphere(point3 cen0, point3 cen1, double t0, double t1, double r, std::shared_ptr<material> m)
		: center0(cen0), center1(cen1), time0(t0), time1(t1), radius(r), mat_ptr(m) {}

	virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const override;

	point3 center(double time) const;

private:
	point3 center0, center1;
	double time0, time1;
	double radius;
	std::shared_ptr<material> mat_ptr;
};

point3 moving_sphere::center(double time) const {
	return center0 + ((time - time0) / (time1 - time0)) * (center1 - center0);
}

bool moving_sphere::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
    vec3 oc = r.origin() - center(r.time());
    double a = r.direction().length_squared();
    double half_b = dot(r.direction(), oc);
    double c = oc.length_squared() - radius * radius;
    double discriminant = half_b * half_b - a * c;

    if (discriminant <= 0) {
        return false;
    }

    double temp = std::sqrt(discriminant);
    double root = (-half_b - temp) / a;
    if (root > t_min && root < t_max) {
        rec.t = root;
        rec.p = r.at(root);
        vec3 outward_normal = (rec.p - center(r.time())) / radius;
        rec.set_front_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;
        return true;
    }

    root = (-half_b + temp) / a;
    if (root > t_min && root < t_max) {
        rec.t = root;
        rec.p = r.at(root);
        vec3 outward_normal = (rec.p - center(r.time())) / radius;
        rec.set_front_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;
        return true;
    }

    return false;
}