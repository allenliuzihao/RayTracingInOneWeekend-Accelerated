// RayTracingTutorial.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>

#include "vec3.h"
#include "color.h"

int main()
{
    const int image_width = 256;
    const int image_height = 256;
    
    std::cout << "P3\n" << image_width << ' ' << image_height << "\n255\n";

    for (int row = image_height - 1; row >= 0; --row) {

        std::cerr << "\nScanlines remaining: " << row + 1 << ' ' << std::flush;
        for (int col = 0; col < image_width; ++col) {
            double r = double(col) / ((double)image_width - 1.0);
            double g = double(row) / ((double)image_height - 1.0);
            double b = 0.25;

            color pixel_color(r, g, b);
            write_color(std::cout, pixel_color);
        }
    }

    std::cerr << "\nDone.\n";
}