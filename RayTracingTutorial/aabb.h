#pragma once

#include "vec3.h"
#include "ray.h"

class aabb {
public:
	aabb() {}
	aabb(const point3& a, const point3& b) { _min = a; _max = b; }

	point3 min() const { return _min; }
	point3 max() const { return _max; }

	bool hit(const ray& r, double tmin, double tmax) const {
		for (int a = 0; a < 3; ++a) {
			auto t0 = std::fmin((_min[a] - r.origin()[a]) / r.direction()[a], (_max[a] - r.origin()[a]) / r.direction()[a]);
			auto t1 = std::fmax((_min[a] - r.origin()[a]) / r.direction()[a], (_max[a] - r.origin()[a]) / r.direction()[a]);

			tmin = std::fmax(t0, tmin);
			tmax = std::fmin(t1, tmax);

			if (tmin >= tmax) {
				return false;
			}
		}

		return true;
	}

private:
	point3 _min;
	point3 _max;
};