#include <iostream>

#include "utilities.h"

#include "color.h"
#include "hittables.h"
#include "sphere.h"
#include "camera.h"

color ray_color(const ray& r, hittable& world, int depth) {
    hit_record rec;

    if (depth <= 0) {
        return color(0, 0, 0);
    }

    if (world.hit(r, 0, infinity, rec)) {
        point3 target = rec.p + rec.normal + random_in_unit_sphere();

        return 0.5 * ray_color(ray(rec.p, target - rec.p), world, depth - 1);
    }

    vec3 unit_dir = unit_vector(r.direction());
    double t = 0.5 * (unit_dir.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main()
{
    const double aspect_ratio = 16.0 / 9.0;
    const int image_width = 600;
    const int image_height = static_cast<int>(image_width / aspect_ratio);
    const int samples_per_pixel = 100;
    const int max_depth = 50;

    hittables world;
    world.add(std::make_shared<sphere>(point3(0, 0, -1), 0.5));
    world.add(std::make_shared<sphere>(point3(0, -100.5, -1), 100));

    camera cam;

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int row = image_height - 1; row >= 0; --row) {

        std::cerr << "\nScanlines remaining: " << row + 1 << ' ' << std::flush;
        for (int col = 0; col < image_width; ++col) {
            color pixel_color(0, 0, 0);
            for (int sample = 0; sample < samples_per_pixel; ++sample) {
                double u = (col * 1.0 + random_double()) / (image_width - 1.0);
                double v = (row * 1.0 + random_double()) / (image_height - 1.0);
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }

    std::cerr << "\nDone.\n";
}