#pragma once

#include <iostream>

#include "vec3.h"

void write_color(std::ostream& out, const color& pixel_color) {
    out << static_cast<int>(255 * pixel_color.x()) << ' '
        << static_cast<int>(255 * pixel_color.y()) << ' '
        << static_cast<int>(255 * pixel_color.z()) << '\n';
}