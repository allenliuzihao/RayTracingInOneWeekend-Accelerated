#include <iostream>
#include <thread>

#include "utilities.h"

#include "color.h"
#include "hittables.h"
#include "sphere.h"
#include "camera.h"

#include "material.h"
#include "lambertian.h"
#include "metal.h"
#include "dialectric.h"

color ray_color(const ray& r, const hittable& world, int depth) {
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

hittables random_scene() {
    hittables world;

    auto ground_material = std::make_shared<lambertian>(color(0.5, 0.5, 0.5));
    world.add(std::make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double();
            point3 center(a + 0.9 * random_double(), 0.2, b + 0.9 * random_double());

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {
                std::shared_ptr<material> sphere_material;

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random() * color::random();
                    sphere_material = std::make_shared<lambertian>(albedo);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                }
                else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = color::random(0.5, 1);
                    auto fuzz = random_double(0, 0.5);
                    sphere_material = std::make_shared<metal>(albedo, fuzz);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                }
                else {
                    // glass
                    sphere_material = std::make_shared<dielectric>(1.5);
                    world.add(std::make_shared<sphere>(center, 0.2, sphere_material));
                }
            }
        }
    }

    auto material1 = std::make_shared<dielectric>(1.5);
    world.add(std::make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

    auto material2 = std::make_shared<lambertian>(color(0.4, 0.2, 0.1));
    world.add(std::make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

    auto material3 = std::make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
    world.add(std::make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

    return world;
}

void render(std::vector<std::vector<vec3>>& image_grid,
            int samples_per_pixel,
            int max_depth,
            const hittables& world,
            const camera& cam,
            const std::pair<int, int> tile_origin, 
            const std::pair<int, int> tile_dim, 
            const std::pair<int, int> image_dim) {
    int row_bound = std::min(image_dim.first , tile_origin.first + tile_dim.first);
    int col_bound = std::min(image_dim.second, tile_origin.second + tile_dim.second);

    for (int row = tile_origin.first; row < row_bound; ++row) {
        for (int col = tile_origin.second; col < col_bound; ++col) {            
            for (int sample = 0; sample < samples_per_pixel; ++sample) {
                double u = (col * 1.0 + random_double()) / image_dim.second;
                double v = (row * 1.0 + random_double()) / image_dim.first;
                ray r = cam.get_ray(u, v);
                image_grid[row][col] += ray_color(r, world, max_depth);
            }
        }
    }
}

int main() {
    // renderer configuration
    auto aspect_ratio = 3.0 / 2.0;
    auto image_width = 1200;
    auto image_height = static_cast<int>(image_width / aspect_ratio);
    auto samples_per_pixel = 500;
    auto max_depth = 50;

    // cpu configurations
    unsigned num_cpus_context = std::thread::hardware_concurrency();
    std::vector<int> factors = find_closest_factors(num_cpus_context);
    int num_tiles_horizontal = factors[0], num_tiles_vertical = factors[1];

    if (image_width < image_height) {
        num_tiles_horizontal = factors[1];
        num_tiles_vertical = factors[0];
    }

    int tile_width = (int) ceil(image_width / num_tiles_horizontal);
    int tile_height = (int) ceil(image_height / num_tiles_vertical);

    auto world = random_scene();
    
    auto material_ground = std::make_shared<lambertian>(color(0.8, 0.8, 0.0));
    auto material_center = std::make_shared<lambertian>(color(0.1, 0.2, 0.5));
    auto material_left = std::make_shared<dielectric>(1.5);
    auto material_right = std::make_shared<metal>(color(0.8, 0.6, 0.2), 0.0);

    world.add(std::make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0, material_ground));
    world.add(std::make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5, material_center));
    world.add(std::make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5, material_left));
    world.add(std::make_shared<sphere>(point3(-1.0, 0.0, -1.0), -0.45, material_left));
    world.add(std::make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5, material_right));

    // Camera
    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    double aperture = 0.1;

    camera cam(lookfrom, lookat, vup, 20.0, aspect_ratio, aperture, dist_to_focus);

    std::vector<std::vector<vec3>> image_grid(image_height);
    for (int row = 0; row < image_height; ++row) {
        image_grid[row].resize(image_width, vec3(0, 0, 0));
    }

    // TODO: parallelize this on cpu cores, with one thread per core.
    for (int row = 0; row < image_height; row += tile_height) {
        for (int col = 0; col < image_width; col += tile_width) {
            render(image_grid,
                   samples_per_pixel, max_depth, 
                   world, cam, 
                   std::make_pair(row, col), 
                   std::make_pair(tile_height, tile_width), 
                   std::make_pair(image_height, image_width));
        }
    }

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int row = image_height - 1; row >= 0; --row) {
        for (int col = 0; col < image_width; ++col) {
            write_color(std::cout, image_grid[row][col], samples_per_pixel);
        }
    }

    std::cerr << "\nDone.\n";
}