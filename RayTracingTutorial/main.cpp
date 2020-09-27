#include <iostream>

#include "utilities.h"

#include "color.h"
#include "hittables.h"
#include "sphere.h"

color ray_color(const ray& r, hittable& world) {
    hit_record rec;
    if (world.hit(r, 0, infinity, rec)) {
        return 0.5 * (rec.normal + color(1, 1, 1));
    }

    vec3 unit_dir = unit_vector(r.direction());
    double t = 0.5 * (unit_dir.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main()
{
    const double aspect_ratio = 16.0 / 9.0;
    const int image_width = 400;
    const int image_height = static_cast<int>(image_width / aspect_ratio);

    hittables world;
    world.add(std::make_shared<sphere>(point3(0, 0, -1), 0.5));
    world.add(std::make_shared<sphere>(point3(0, -100.5, -1), 100));

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
            color pixel_color = ray_color(r, world);
            write_color(std::cout, pixel_color);
        }
    }

    std::cerr << "\nDone.\n";
}