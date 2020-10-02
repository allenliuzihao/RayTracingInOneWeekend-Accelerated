#pragma once

#include "utilities.h"

class camera {
public:
	camera(point3 lookfrom,
		   point3 lookat,
		   vec3   vup,
		   double fov, 
		   double aspect_ratio,
	       double focal_length) {
		
		double theta = degrees_to_radians(fov);
		double h = focal_length * tan(theta / 2.0);

		double viewport_height = 2.0 * h;
		double viewport_width = aspect_ratio * viewport_height;

		auto w = unit_vector(lookfrom - lookat);
		auto u = unit_vector(cross(vup, w));
		auto v = cross(w, u);

		origin = lookfrom;
		horizontal = viewport_width * u;
		vertical = viewport_height * v;
		lower_left_corner = origin - horizontal/2.0 - vertical/2.0 - focal_length * w;
	}

	ray get_ray(double s, double t) const {
		return ray(origin, lower_left_corner + s * horizontal + t * vertical - origin);
	}

private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
};