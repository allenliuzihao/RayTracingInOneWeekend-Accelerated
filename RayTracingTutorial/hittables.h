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

	const std::vector<std::shared_ptr<hittable>>& getObjects() const { return objects; }

	virtual bool hit(const ray& r, double tmin, double tmax, hit_record& rec) const override;
	virtual bool bounding_box(double t0, double t1, aabb& output_box) const override;
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

bool hittables::bounding_box(double t0, double t1, aabb& output_box) const {
	if (objects.empty()) {
		return false;
	}

	aabb temp_box;
	bool first_box = true;

	for (const auto& object : objects) {
		if (!object->bounding_box(t0, t1, temp_box)) {
			return false;
		}
		output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
		first_box = false;
	}

	return true;
}