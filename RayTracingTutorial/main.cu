#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include "utilities.h"

#include "color.h"
#include "hittables.h"
#include "sphere.h"
#include "camera.h"

#include "material.h"
#include "lambertian.h"
#include "metal.h"
#include "dialectric.h" 

__constant__ camera cam;

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

hittables random_scene(hittables* d_world, cudaStream_t stream) {
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

void init_host_image_buffer(color* image_buffer, int image_width, int image_height) {
    for (int row = 0; row < image_height; ++row) {
        for (int col = 0; col < image_width; ++col) {
            image_buffer[row * image_width + col] = color(0, 0, 0);
        }
    }
}

int main() {
    // initialize render config
    auto aspect_ratio = 3.0 / 2.0;
    auto image_width = 1200;
    auto image_height = static_cast<int>(image_width / aspect_ratio);
    auto samples_per_pixel = 500;
    auto max_depth = 50;

    // initialize camera
    point3 lookfrom(13, 2, 3);
    point3 lookat(0, 0, 0);
    vec3 vup(0, 1, 0);
    auto dist_to_focus = 10.0;
    double aperture = 0.1;
    camera host_cam(lookfrom, lookat, vup, 20.0, aspect_ratio, aperture, dist_to_focus);


    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    unsigned int size_image_buffer = image_width * image_height;
    unsigned int mem_size_image_buffer = size_image_buffer * sizeof(color);
    color* image_buffer, * d_image_buffer;
    hittables* d_world;
    cudaStream_t stream_image_buffer, stream_camera, stream_world;

    checkCudaErrors(cudaStreamCreate(&stream_image_buffer));
    checkCudaErrors(cudaStreamCreate(&stream_camera));
    checkCudaErrors(cudaStreamCreate(&stream_world));

    // copy camera data over to device memory
    checkCudaErrors(cudaMemcpyToSymbolAsync(cam, &host_cam, sizeof(camera), 0, cudaMemcpyDefault, stream_camera));

    // TODO: initialize world


    // copy image data buffer to device memory
    std::cerr << "Image width: " << image_width << " image height: " << image_height << "\n";
    std::cerr << "Allocating " << size_image_buffer << " number of pixels with " << mem_size_image_buffer << " bytes on host and device.\n";

    checkCudaErrors(cudaMallocHost(&image_buffer, mem_size_image_buffer));
    init_host_image_buffer(image_buffer, image_width, image_height);

    checkCudaErrors(cudaMalloc(&d_image_buffer, mem_size_image_buffer));
    checkCudaErrors(cudaMemcpyAsync(d_image_buffer, image_buffer, mem_size_image_buffer, cudaMemcpyHostToDevice, stream_image_buffer));


    // wait for render initialization to finish
    checkCudaErrors(cudaStreamSynchronize(stream_image_buffer));
    checkCudaErrors(cudaStreamSynchronize(stream_camera));
    checkCudaErrors(cudaStreamSynchronize(stream_world));

    /*
    for (int row = image_height - 1; row >= 0; --row) {

        std::cerr << "\nScanlines remaining: " << row + 1 << ' ' << std::flush;
        for (int col = 0; col < image_width; ++col) {
            color pixel_color(0, 0, 0);
            for (int sample = 0; sample < samples_per_pixel; ++sample) {
                double u = (col * 1.0 + random_double()) / image_width;
                double v = (row * 1.0 + random_double()) / image_height;
                ray r = cam.get_ray(u, v);
                pixel_color += ray_color(r, world, max_depth);
            }
            write_color(std::cout, pixel_color, samples_per_pixel);
        }
    }
    */

    checkCudaErrors(cudaMemcpyAsync(image_buffer, d_image_buffer, mem_size_image_buffer, cudaMemcpyDeviceToHost, stream_image_buffer));
    checkCudaErrors(cudaStreamSynchronize(stream_image_buffer));

    std::cerr << "Writing result from device to host\n";

    for (int row = image_height - 1; row >= 0; --row) {
        for (int col = 0; col < image_width; ++col) {
            //write_color(std::cout, image_buffer[row * image_width + col], samples_per_pixel);
        }
    }

    std::cerr << "\nDone.\n";

    checkCudaErrors(cudaFreeHost(image_buffer));
    checkCudaErrors(cudaFree(d_image_buffer));
    checkCudaErrors(cudaFree(d_world));
}