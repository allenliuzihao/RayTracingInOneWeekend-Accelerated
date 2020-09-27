#pragma once

#include "hittable.h"

#include <memory>
#include <vector>

class hittables : public hittable {
public:
	hittables() {}
	hittables(std::shared_ptr<hittable> object) { add(object); };

	void clear() { objects.clear(); }
	void add(std::shared_ptr<hittable> object) { objects.push_back(object); };

	virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const override;

private:
	std::vector<std::shared_ptr<hittable>> objects;
};

bool hittables::hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
	hit_record temp;
	bool hit_anything = false;
	double closest_hit_so_far = t_max;

	for (const auto& object : objects) {
		if (object->hit(r, t_min, closest_hit_so_far, temp)) {
			closest_hit_so_far = temp.t;
			rec = temp;
			hit_anything = true;
		}
	}

	return hit_anything;
}