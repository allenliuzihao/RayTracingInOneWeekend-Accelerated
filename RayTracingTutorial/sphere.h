#pragma once

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
public:
	sphere() : center({0.0, 0.0, 0.0}), radius(0.0) {}
	sphere(point3 cen, double r, std::shared_ptr<material> m) : center(cen), radius(r), mat_ptr(m) {}
	
	virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const override;
    virtual bool bounding_box(double t0, double t1, aabb& output_box) const override;
private:
	point3 center;
	double radius;
    std::shared_ptr<material> mat_ptr;
};

void get_sphere_uv(const vec3& p, double& u, double& v) {
    auto phi = atan2(p.y(), p.x());
    auto theta = asin(p.z());
    u = 1 - (phi + pi) / (2 * pi);
    v = (theta + pi / 2) / pi;
}

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
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_front_normal(r, outward_normal);
        get_sphere_uv((rec.p - center) / radius , rec.u, rec.v);
        rec.mat_ptr = mat_ptr;
        return true;
    }

    root = (-half_b + temp) / a;
    if (root > tmin && root < tmax) {
        rec.t = root;
        rec.p = r.at(root);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_front_normal(r, outward_normal);
        get_sphere_uv((rec.p - center) / radius, rec.u, rec.v);
        rec.mat_ptr = mat_ptr;
        return true;
    }

    return false;
}

bool sphere::bounding_box(double t0, double t1, aabb& output_box) const {
    output_box = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
    return true;
}