#pragma once

#include "hittable.h"
#include "vec3.h"

class sphere : public hittable {
public:
    __device__ sphere(point3 cen, double r, material* m_ptr) : center(cen), radius(r), mat_ptr(m_ptr) {}
	
    __device__ virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const override;

    __device__ material* get_mat_ptr() { return mat_ptr; }

private:
	point3 center;
	double radius;
    material* mat_ptr;
};

__device__ bool sphere::hit(const ray& r, double tmin, double tmax, hit_record& rec) const {
    vec3 oc = r.origin() - center;
    double a = r.direction().length_squared();
    double half_b = dot(r.direction(), oc);
    double c = oc.length_squared() - radius * radius;
    double discriminant = half_b * half_b - a * c;

    if (discriminant <= 0) {
        return false;
    }

    double temp = sqrt(discriminant);
    double root = (-half_b - temp) / a;
    if (root > tmin && root < tmax) {
        rec.t = root;
        rec.p = r.at(root);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_front_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;
        return true;
    }

    root = (-half_b + temp) / a;
    if (root > tmin && root < tmax) {
        rec.t = root;
        rec.p = r.at(root);
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_front_normal(r, outward_normal);
        rec.mat_ptr = mat_ptr;
        return true;
    }

    return false;
}