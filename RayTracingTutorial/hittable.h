#pragma once

#include "aabb.h"
#include "utilities.h"
#include "ray.h"

class material;

struct hit_record {
	point3 p;
	vec3 normal;
	double t;
	bool front_face;
	std::shared_ptr<material> mat_ptr;

	inline void set_front_normal(const ray & r, const vec3& outward_normal) {
		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};

class hittable {
public:
	virtual bool hit(const ray & r, double t_min, double t_max, hit_record & rec) const = 0;
	virtual bool bounding_box(double t0, double t1, aabb& output_box) const = 0;
};