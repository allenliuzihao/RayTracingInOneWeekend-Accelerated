#include <iostream>

#include "utilities.h"

#include "color.h"
#include "hittables.h"
#include "sphere.h"
#include "camera.h"

#include "material.h"
#include "lambertian.h"
#include "metal.h"
#include "dialectric.h"

color ray_color(const ray& r, hittable& world, int depth) {
    hit_record rec;

    if (depth <= 0) {
        return color(0, 0, 0);
    }

    if (world.hit(r, 0.001, infinity, rec)) {
        ray scatter;
        color attenuation;
        if (rec.mat_ptr->scatter(r, rec, attenuation, scatter)) {
            return attenuation * ray_color(scatter, world, depth - 1);
        }
        return color(0, 0, 0);
    }

    vec3 unit_dir = unit_vector(r.direction());
    double t = 0.5 * (unit_dir.y() + 1.0);
    return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

int main()
{
    auto aspect_ratio = 16.0 / 9.0;
    auto image_width = 600;
    auto image_height = static_cast<int>(image_width / aspect_ratio);
    auto samples_per_pixel = 100;
    auto max_depth = 50;

    auto R = cos(pi / 4.0);
    hittables world;
    
    auto material_left = std::make_shared<lambertian>(color(0,0,1));
    auto material_right = std::make_shared<lambertian>(color(1,0,0));

    world.add(std::make_shared<sphere>(point3(-R, 0.0, -1.0), R, material_left));
    world.add(std::make_shared<sphere>(point3(R, 0.0, -1.0), R, material_right));

    // Camera
    camera cam(90.0, aspect_ratio);

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