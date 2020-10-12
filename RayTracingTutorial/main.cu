#include <iostream>

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <helper_timer.h>

#include "utilities.h"

#include "color.h"
#include "hittables.h"
#include "sphere.h"
#include "camera.h"

#include "material.h"
#include "lambertian.h"
#include "metal.h"
#include "dialectric.h" 

const int MAX_GPU_COUNT = 2;

__constant__ camera cam[MAX_GPU_COUNT];

typedef struct {
    color* d_image_buffer;
    color* h_image_buffer;
    size_t mem_size_image_buffer;

    int width_from;
    int width_to;
    int height_from;
    int height_to;

    hittable** d_objects;
    hittable** d_world;

    curandState* d_rand_state_create_world;
    curandState* d_rand_state_render;

    cudaStream_t stream_image_buffer;
    cudaStream_t stream_camera;
    cudaStream_t stream_world;
} GPUPlan;

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

__global__ void render_init(int image_width_index, int image_width, int image_height, curandState* rand_state) {
    int i = image_width_index + threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= image_width || j >= image_height) {
        return;
    }

    int pixel_index = j * image_width + i;
    curand_init(1993, pixel_index, 0, &rand_state[pixel_index]);
}

__global__ void render(int gpu, color* image_buffer, int image_width_index, int image_width, int image_height, int samples_per_pixel, int max_depth, hittable** world, curandState* rand_state) {
    int i = image_width_index + threadIdx.x + blockIdx.x * blockDim.x;
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
        ray r = cam[gpu].get_ray(u, v, &local_rand_state);
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

void init_host_image_buffers(GPUPlan gpus[], int NUM_GPUs, int image_width, int image_height) {
    int delta_width = image_width / NUM_GPUs;
    
    int width_from = 0, width_to = width_from + delta_width;
    unsigned int curr_width, size_image_buffer;

    for (int i = 0; i < NUM_GPUs; ++i) {
        if (i == NUM_GPUs - 1) {
            width_to = image_width;
        }

        curr_width = width_to - width_from;
        size_image_buffer = curr_width * image_height;
        gpus[i].mem_size_image_buffer = size_image_buffer * sizeof(color);
        gpus[i].width_from = width_from;
        gpus[i].width_to = width_to;
        gpus[i].height_from = 0;
        gpus[i].height_to = image_height;

        checkCudaErrors(cudaMallocHost(&gpus[i].h_image_buffer, gpus[i].mem_size_image_buffer));
        
        for (int row = 0; row < image_height; ++row) {
            for (int col = 0; col < curr_width; ++col) {
                gpus[i].h_image_buffer[row * curr_width + col] = color(0, 0, 0);
            }
        }

        width_from = width_to;
        width_to = width_from + delta_width;
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

    // world
    int num_hittables = 485;

    // GPUs
    int NUM_GPUS = MAX_GPU_COUNT;
    GPUPlan gpus[MAX_GPU_COUNT];
    
    checkCudaErrors(cudaGetDeviceCount(&NUM_GPUS));
    NUM_GPUS = std::min(NUM_GPUS, MAX_GPU_COUNT);
    
    std::cerr << "CUDA-capable device count: " << NUM_GPUS << "%i\n";

    // images
    init_host_image_buffers(gpus, NUM_GPUS, image_width, image_height);

    for (int i = 0; i < NUM_GPUS; i++) {
        checkCudaErrors(cudaSetDevice(i));

        checkCudaErrors(cudaStreamCreate(&gpus[i].stream_image_buffer));
        checkCudaErrors(cudaStreamCreate(&gpus[i].stream_camera));
        checkCudaErrors(cudaStreamCreate(&gpus[i].stream_world));

        checkCudaErrors(cudaMemcpyToSymbolAsync(cam[i], &host_cam, sizeof(camera), 0, cudaMemcpyDefault, gpus[i].stream_camera));
        
        checkCudaErrors(cudaMalloc(&gpus[i].d_rand_state_create_world, sizeof(curandState)));
        rand_init <<<1, 1, 0, gpus[i].stream_world>>> (gpus[i].d_rand_state_create_world);
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMalloc(&gpus[i].d_objects, num_hittables * sizeof(hittable*)));
        checkCudaErrors(cudaMalloc(&gpus[i].d_world, sizeof(hittable*)));

        random_scene <<<1, 1, 0, gpus[i].stream_world>>> (gpus[i].d_objects, gpus[i].d_world, gpus[i].d_rand_state_create_world);

        checkCudaErrors(cudaMalloc(&gpus[i].d_image_buffer, gpus[i].mem_size_image_buffer));
        checkCudaErrors(cudaMemcpyAsync(gpus[i].d_image_buffer, gpus[i].h_image_buffer, gpus[i].mem_size_image_buffer, cudaMemcpyHostToDevice, gpus[i].stream_image_buffer));
    }

    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    StopWatchInterface* timer = NULL;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    int curr_image_width, curr_image_height = image_height;
    int curr_image_width_index = 0;
    dim3 threads_per_block(16, 16);
    dim3 blocks_per_grid;

    for (int i = 0; i < NUM_GPUS; ++i) {
        checkCudaErrors(cudaSetDevice(i));

        curr_image_width = gpus[i].width_to - gpus[i].width_from;
        blocks_per_grid.x = (curr_image_width + threads_per_block.x - 1) / threads_per_block.x;
        blocks_per_grid.y = (curr_image_height + threads_per_block.y - 1) / threads_per_block.y;

        checkCudaErrors(cudaMalloc(&gpus[i].d_rand_state_render, curr_image_width * curr_image_height * sizeof(curandState)));

        render_init <<<blocks_per_grid, threads_per_block, 0, gpus[i].stream_image_buffer>>> (curr_image_width_index, image_width, image_height, gpus[i].d_rand_state_render);
        checkCudaErrors(cudaGetLastError());

        render <<<blocks_per_grid, threads_per_block, 0, gpus[i].stream_image_buffer>>> (i, gpus[i].d_image_buffer, curr_image_width_index, image_width, image_height, samples_per_pixel, max_depth, gpus[i].d_world, gpus[i].d_rand_state_render);
        checkCudaErrors(cudaGetLastError());

        checkCudaErrors(cudaMemcpyAsync(gpus[i].h_image_buffer, gpus[i].d_image_buffer, gpus[i].mem_size_image_buffer, cudaMemcpyDeviceToHost, gpus[i].stream_image_buffer));

        curr_image_width_index += curr_image_width;
    }
    
    for (int i = 0; i < NUM_GPUS; i++) {
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }

    sdkStopTimer(&timer);
    std::cerr << "\nrendering complete.\n GPU time used: " << sdkGetTimerValue(&timer) << " ms\n";

    std::cerr << "Writing result from device to host\n";

    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    int curr_block_width;
    for (int row = image_height - 1; row >= 0; --row) {
        for (int i = 0; i < NUM_GPUS; ++i) {
            curr_block_width = gpus[i].width_to - gpus[i].width_from;
            for (int col = 0; col < curr_block_width; ++col) {
                write_color(std::cout, gpus[i].h_image_buffer[row * curr_block_width + col], samples_per_pixel);
            }
        }
    }

    std::cerr << "\nDone.\n";

    sdkDeleteTimer(&timer);

    for (int i = 0; i < NUM_GPUS; ++i) {
        checkCudaErrors(cudaSetDevice(i));

        free_world <<<1, 1, 0, gpus[i].stream_world>>> (gpus[i].d_objects, num_hittables, gpus[i].d_world);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaStreamSynchronize(gpus[i].stream_world));

        checkCudaErrors(cudaStreamDestroy(gpus[i].stream_world));
        checkCudaErrors(cudaStreamDestroy(gpus[i].stream_camera));
        checkCudaErrors(cudaStreamDestroy(gpus[i].stream_image_buffer));
        
        checkCudaErrors(cudaFreeHost(gpus[i].h_image_buffer));
        checkCudaErrors(cudaFree(gpus[i].d_image_buffer));

        checkCudaErrors(cudaFree(gpus[i].d_rand_state_create_world));
        checkCudaErrors(cudaFree(gpus[i].d_rand_state_render));

        checkCudaErrors(cudaFree(gpus[i].d_objects));
        checkCudaErrors(cudaFree(gpus[i].d_world));
    }
}