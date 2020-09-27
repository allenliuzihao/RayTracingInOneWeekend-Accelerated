// RayTracingTutorial.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "vec3.h"
#include "color.h"
#include "ray.h"

double hit_sphere(const point3& center, double radius, const ray& r) {
    vec3 oc = r.origin() - center;
    double a = r.direction().length_squared();
    double half_b = dot(r.direction(), oc);
    double c = oc.length_squared() - radius * radius;
    double discriminant = half_b * half_b - a * c;

    if (discriminant < 0) {
        return -1;
    }
    return (-half_b - std::sqrt(discriminant)) / a;
}

color ray_color(const ray& r) {
    point3 circle_center = point3(0, 0, -1);
    double t = hit_sphere(circle_center, 0.5, r);
    if (t > 0.0) {
        point3 hit_point = r.at(t);
        vec3 hit_point_normal = unit_vector(hit_point - circle_center);
        return (hit_point_normal + 1.0) * 0.5;
    }

    vec3 unit_dir = unit_vector(r.direction());
    t = 0.5 * (unit_dir.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main()
{
    const double aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    double viewport_height = 2.0;
    double view_port_width = aspect_ratio * viewport_height;
    double focal_length = 1.0;

    point3 origin = point3(0, 0, 0);
    vec3 horizontal = vec3(view_port_width, 0, 0);
    vec3 vertical = vec3(0, viewport_height, 0);

    vec3 lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - vec3(0, 0, focal_length);

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int row = image_height - 1; row >= 0; --row) {

        std::cerr << "\nScanlines remaining: " << row + 1 << ' ' << std::flush;
        for (int col = 0; col < image_width; ++col) {
            double u = double(col) / ((double)image_width - 1.0);
            double v = double(row) / ((double)image_height - 1.0);

            ray r(origin,  lower_left_corner + u * horizontal + v * vertical);
            color pixel_color = ray_color(r);
            write_color(std::cout, pixel_color);
        }
    }

    std::cerr << "\nDone.\n";
}