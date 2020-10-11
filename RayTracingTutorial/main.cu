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

__device__ color ray_color(const ray& r, hittable** world, int depth, curandState* rand_state) {
    hit_record rec;

    if (depth <= 0) {
        return color(0, 0, 0);
    }

    ray curr_ray = r;
    color curr_attenuation(1, 1, 1);

    for (int i = 0; i < depth; ++i) {
        if ((*world)->hit(curr_ray, 0.001, infinity, rec)) {
            ray scatter;
            color attenuation;
            if (rec.mat_ptr->scatter(curr_ray, rec, attenuation, scatter, rand_state)) {
                curr_ray = scatter;
                curr_attenuation *= attenuation;
            } else {
                return color(0, 0, 0);
            }
        } else {
            vec3 unit_dir = unit_vector(curr_ray.direction());
            double t = 0.5 * (unit_dir.y() + 1.0);
            return curr_attenuation * ((1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0));
        }
    }
    return color(0.0, 0.0, 0.0);
}

__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1993, 0, 0, rand_state);
    }
}

__global__ void free_world(hittable** d_objects, int num_hittables, hittable** d_world) {
    for (int i = 0; i < num_hittables; ++i) {
        delete ((sphere*)d_objects[i])->get_mat_ptr();
        delete d_objects[i];
    }
    delete *d_world;
}

__global__ void random_scene(hittable** d_objects, hittable** d_world, curandState* rand_state) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    curandState local_rand_state = *rand_state;
    d_objects[0] = new sphere(point3(0, -1000, 0), 1000, new lambertian(color(0.5, 0.5, 0.5)));

    int i = 1;

    for (int a = -11; a < 11; a++) {
        for (int b = -11; b < 11; b++) {
            auto choose_mat = random_double(&local_rand_state);
            point3 center(a + 0.9 * random_double(&local_rand_state), 0.2, b + 0.9 * random_double(&local_rand_state));

            if ((center - point3(4, 0.2, 0)).length() > 0.9) {

                if (choose_mat < 0.8) {
                    // diffuse
                    auto albedo = color::random(&local_rand_state) * color::random(&local_rand_state);
                    d_objects[i++] = new sphere(center, 0.2, new lambertian(albedo));
                } else if (choose_mat < 0.95) {
                    // metal
                    auto albedo = (color::random(&local_rand_state) + 1.0) * 0.5;
                    auto fuzz = random_double(&local_rand_state) * 0.5;
                    d_objects[i++] = new sphere(center, 0.2, new metal(albedo, fuzz));
                } else {
                    // glass
                    d_objects[i++] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }
    }

    d_objects[i++] = new sphere(point3(0, 1, 0), 1.0, new dielectric(1.5));
    d_objects[i++] = new sphere(point3(-4, 1, 0), 1.0, new lambertian(color(0.4, 0.2, 0.1)));
    d_objects[i++] = new sphere(point3(4, 1, 0), 1.0, new metal(color(0.7, 0.6, 0.5), 0.0));

    *rand_state = local_rand_state;
    *d_world = (hittable*) new hittables(d_objects, i);
}

__global__ void render_init(int image_width, int image_height, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= image_width || j >= image_height) {
        return;
    }

    int pixel_index = j * image_width + i;
    curand_init(1993, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(color* image_buffer, int image_width, int image_height, int samples_per_pixel, int max_depth, hittable** world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= image_width || j >= image_height) {
        return;
    }

    int pixel_index = j * image_width + i;
    curandState local_rand_state = rand_state[pixel_index];

    color pixel_color(0, 0, 0);
    for (int sample = 0; sample < samples_per_pixel; ++sample) {
        double u = (i * 1.0 + random_double(&local_rand_state)) / image_width;
        double v = (j * 1.0 + random_double(&local_rand_state)) / image_height;
        ray r = cam.get_ray(u, v, &local_rand_state);
        pixel_color += ray_color(r, world, max_depth, &local_rand_state);
    }

    image_buffer[pixel_index] = pixel_color;
    rand_state[pixel_index] = local_rand_state;
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
    auto image_width = 200;         // 1200
    auto image_height = static_cast<int>(image_width / aspect_ratio);
    auto samples_per_pixel = 10;    // 500
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
    cudaStream_t stream_image_buffer, stream_camera, stream_world;

    checkCudaErrors(cudaStreamCreate(&stream_image_buffer));
    checkCudaErrors(cudaStreamCreate(&stream_camera));
    checkCudaErrors(cudaStreamCreate(&stream_world));

    // copy camera data over to device memory
    checkCudaErrors(cudaMemcpyToSymbolAsync(cam, &host_cam, sizeof(camera), 0, cudaMemcpyDefault, stream_camera));

    // initialize world
    curandState* d_rand_state_create_world;
    checkCudaErrors(cudaMalloc(&d_rand_state_create_world, sizeof(curandState)));

    rand_init <<<1, 1, 0, stream_world>>> (d_rand_state_create_world);
    checkCudaErrors(cudaGetLastError());

    hittable** d_objects;
    hittable** d_world;
    int num_hittables = 485;
    checkCudaErrors(cudaMalloc(&d_objects, num_hittables * sizeof(hittable*)));
    checkCudaErrors(cudaMalloc(&d_world, sizeof(hittable*)));

    random_scene <<<1,1,0,stream_world>>>(d_objects, d_world, d_rand_state_create_world);

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

    // prepare rendering with a curand state per pixel
    dim3 threads_per_block(8, 8);
    dim3 blocks_per_grid((image_width + threads_per_block.x - 1) / threads_per_block.x, (image_height + threads_per_block.y - 1) / threads_per_block.y);

    curandState* d_rand_state_render;
    checkCudaErrors(cudaMalloc(&d_rand_state_render, image_width * image_height * sizeof(curandState)));

    render_init <<<blocks_per_grid, threads_per_block, 0, stream_image_buffer>>> (image_width, image_height, d_rand_state_render);
    checkCudaErrors(cudaGetLastError());

    render <<<blocks_per_grid, threads_per_block, 0, stream_image_buffer>>> (d_image_buffer, image_width, image_height, samples_per_pixel, max_depth, d_world, d_rand_state_render);
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpyAsync(image_buffer, d_image_buffer, mem_size_image_buffer, cudaMemcpyDeviceToHost, stream_image_buffer));
    checkCudaErrors(cudaStreamSynchronize(stream_image_buffer));

    std::cerr << "Writing result from device to host\n";

    for (int row = image_height - 1; row >= 0; --row) {
        for (int col = 0; col < image_width; ++col) {
            write_color(std::cout, image_buffer[row * image_width + col], samples_per_pixel);
        }
    }

    std::cerr << "\nDone.\n";

    free_world <<<1, 1, 0, stream_world>>> (d_objects, num_hittables, d_world);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaStreamSynchronize(stream_world));

    checkCudaErrors(cudaStreamDestroy(stream_world));
    checkCudaErrors(cudaStreamDestroy(stream_camera));
    checkCudaErrors(cudaStreamDestroy(stream_image_buffer));

    checkCudaErrors(cudaFreeHost(image_buffer));
    checkCudaErrors(cudaFree(d_image_buffer));

    checkCudaErrors(cudaFree(d_rand_state_create_world));
    checkCudaErrors(cudaFree(d_rand_state_render));
    checkCudaErrors(cudaFree(d_objects));
    checkCudaErrors(cudaFree(d_world));
}