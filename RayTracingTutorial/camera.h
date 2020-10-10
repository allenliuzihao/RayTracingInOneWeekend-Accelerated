#pragma once

#include "utilities.h"

class camera {
public:
	__device__ camera() {}

	camera(point3 lookfrom,
					  point3 lookat,
					  vec3   vup,
					  double fov, 
					  double aspect_ratio,
					  double aperture,
					  double focal_dist) {
		
		double theta = degrees_to_radians(fov);
		double h = focal_dist * tan(theta / 2.0);

		double viewport_height = 2.0 * h;
		double viewport_width = aspect_ratio * viewport_height;

		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);

		origin = lookfrom;
		horizontal = viewport_width * u;
		vertical = viewport_height * v;
		lower_left_corner = origin - horizontal/2.0 - vertical/2.0 - focal_dist * w;

		lens_radius = aperture / 2.0f;
	}

	__device__ ray get_ray(double s, double t, curandState *curandState) const {
		vec3 sampled_and_scaled_point = lens_radius * random_in_unit_disk(curandState);
		vec3 offset = sampled_and_scaled_point.x() * u + sampled_and_scaled_point.y() * v;

		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
	}

private:
	point3 origin;
	point3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	double lens_radius;
};