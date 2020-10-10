#pragma once

#include "utilities.h"
#include "hittable.h"

class material {
public:
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, color & attenuation, ray& scattered) const = 0;
};

__device__ double schlick(double cosine, double ref_index) {
    double r0 = (1 - ref_index) / (1 + ref_index);
    r0 = r0 * r0;
    return r0 + (1 - r0) * pow((1 - cosine), 5);
}