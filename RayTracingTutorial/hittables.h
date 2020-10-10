#pragma once

#include "hittable.h"

#include <memory>
#include <vector>

class hittables : public hittable {
public:
	__device__ hittables(hittable** objs, int n) : objects(objs), num_objects(n) {  };

	__device__ void clear() { num_objects = 0; }

	__device__ virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const override;

private:
	int num_objects;
	hittable** objects;
};

__device__ bool hittables::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	hit_record temp;
	bool hit_anything = false;
	double closest_hit_so_far = t_max;

	for (int i = 0; i < num_objects; ++i) {
		if (objects[i]->hit(r, t_min, closest_hit_so_far, temp)) {
			closest_hit_so_far = temp.t;
			rec = temp;
			hit_anything = true;
		}
	}

	return hit_anything;
}