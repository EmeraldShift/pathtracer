#include "tracer.h"

#include "scene/geometry.h"
#include "scene/material.h"
#include "scene/scene.h"
#include "scene/ray.h"

#include <glm/vec3.hpp>
#include <glm/gtx/norm.hpp>
#include <math.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>

constexpr int THREADS_PER_BLOCK = 96;

#define abs(a) (a < 0 ? -a : a)

struct Pair {
    glm::dvec3 a, b;

    __host__ __device__ Pair(glm::dvec3 a, glm::dvec3 b) : a(a), b(b) {}
};

__device__ static Pair
getBasis(glm::dvec3 normal) {
    auto a = abs(normal[0]) > RAY_EPSILON
             ? glm::normalize(glm::cross(glm::dvec3(0, 1, 0), normal))
             : glm::normalize(glm::cross(glm::dvec3(1, 0, 0), normal));
    auto b = glm::cross(normal, a);
    return Pair(a, b);
}

__device__ static glm::dvec3
randomVecFromHemisphere(glm::dvec3 normal, curandState &state) {
    auto basis = getBasis(normal);
    auto p = 2 * M_PI * curand_uniform_double(&state);
    auto cos_p = std::cos(p);
    auto sin_p = std::sin(p);
    auto cos_t = std::pow(curand_uniform_double(&state), 2);
    auto sin_t = std::sqrt(1.0 - cos_t * cos_t);
    return sin_t * cos_p * basis.a + sin_t * sin_p * basis.b + cos_t * normal;
}

/**
 * An iterative version of the CPU ray tracing algorithm. For up to depth
 * iterations, the ray is intersected with the scene, and a random fate
 * is selected. The ray can either reflect diffusely or semi-diffusely,
 * or it can be transmitted through the object.
 *
 * Despite the name, this function traces an entire path of rays, rooted
 * at the given ray.
 *
 * @param scene The scene used for intersection testing
 * @param r The initial ray to trace
 * @param depth The maximum number of iterations to trace
 * @param state The library-provided PRNG state for
 * random number generation
 * @return
 */
__device__ static glm::dvec3
traceRay(Scene *scene, ray &r, int depth, curandState &state) {
    auto color = glm::dvec3(0);
    auto thresh = glm::dvec3(1);
    while (true) {
        isect i;
        if (glm::length2(thresh) < 3.0 * 12.0 / 255.0 / 255.0)
            break;

        // Stop tracing if we miss the scene
        if (!scene->intersectIterative(r, i)) {
            color += thresh;
            break;
        }

        auto hitInner = r.getPosition() + (i.getT() + RAY_EPSILON) * r.getDirection();
        auto hitOuter = r.getPosition() + (i.getT() - RAY_EPSILON) * r.getDirection();

        // Stop tracing if we exceed depth limit
        if (depth < 0) {
            color *= thresh * i.getMaterial().ke(i) * 32.0;
            break;
        }

        glm::dvec3 rad(0);
        auto diffuse = glm::length(i.getMaterial().kd(i));
        auto reflect = i.getMaterial().kr(i).x;
        auto refract = i.getMaterial().kt(i).x;
        auto rand = curand_uniform_double(&state);

        // New ray generation, either refractive or diffuse
        auto refl = r.getDirection() - 2.0 * i.getN() * glm::dot(i.getN(), r.getDirection());
        if (rand < refract) {
            bool into = glm::dot(r.getDirection(), i.getN()) < 0;
            auto n = into ? i.getN() : -i.getN();
            auto n1 = 1.0;
            auto n2 = i.getMaterial().index(i);
            auto ratio = into ? n1 / n2 : n2 / n1;
            auto dot = glm::dot(r.getDirection(), n);
            auto cos2t = 1 - ratio * ratio * (1 - dot * dot);
            if (cos2t < 0) {
                r = ray(hitOuter, refl);
                color += thresh * i.getMaterial().ke(i) * 32.0;
                thresh *= i.getMaterial().kr(i) / 1.0;
                depth--;
            } else {
                auto dir = glm::normalize(r.getDirection() * ratio - n * (dot * ratio + sqrt(cos2t)));
                double a = n2 - n1;
                double b = n2 + n1;
                double R0 = (a * a) / (b * b);
                double c = 1.0 - (into ? -dot : glm::dot(dir, -n));
                double Re = R0 + (1.0 - R0) * c * c * c * c * c;
                double ratio2 = ratio * ratio;
                double Tr = (1.0 - Re) * ratio2;

                double prob = 0.25 + 0.5 * Re;
                // XXX depth test
                r = ray(hitInner, dir);
                color += thresh * i.getMaterial().ke(i) * 32.0;
                thresh *= Tr;
                depth--;
            }
        } else {
            auto basis = getBasis(i.getN());
            auto diff = randomVecFromHemisphere(i.getN(), state);
            auto dir = glm::normalize(reflect * refl + (1 - reflect) * diff);
            r = ray(hitOuter, dir);
            color += thresh * i.getMaterial().ke(i) * 32.0;
            thresh *= i.getMaterial().kd(i);
            depth--;
        }
    }
    return color;
}

__global__ static void
tracePixel(Scene *scene, glm::dvec3 *buffer, unsigned width, unsigned height, unsigned depth, unsigned samples) {
    unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned x = idx % width;
    unsigned y = idx / width;
    if (idx < width * height) {
        curandState state;
        curand_init((unsigned long long) clock() + idx, 0, 0, &state);
        auto sum = glm::dvec3();

        // Trace a ray for each subpixel, add up, and normalize.
        for (auto sx = 0; sx < samples; sx++) {
            for (auto sy = 0; sy < samples; sy++) {
                auto xx = double(x) - 0.5 + (1 + 2 * sx) / (2.0 * samples);
                auto yy = double(y) - 0.5 + (1 + 2 * sy) / (2.0 * samples);
                ray r(glm::dvec3(0, 0, 0), glm::dvec3(0, 0, 0));
                scene->getCamera().rayThrough(xx / double(width), yy / double(height), r);
                auto d = depth;
                sum += glm::clamp(traceRay(scene, r, depth, state), 0.0, 1.0);
            }
        }
        sum /= ((double) samples * samples);
        buffer[idx] = sum;
    }
}

void GpuTracer::traceImage(int width, int height) {
    // Allocate a buffer of color vectors per pixel
    auto size = width * height;
    auto raw = new glm::dvec3[size];
    glm::dvec3 *d_raw;
    cudaMalloc((void **) &d_raw, size * sizeof(glm::dvec3));

    // Does a lot of work--copies the entire scene and returns
    // a pointer to the scene in device memory.
    // See: Scene::clone, BoundedVolumeHierarchy::clone,
    // Cluster::clone, and Geometry(& Sphere/Trimesh)::clone.
    auto d_scene = scene->clone();

    // Yeet the pixel buffer to the device, render, and fetch the values.
    cudaMemcpy(d_raw, raw, size * sizeof(glm::dvec3), cudaMemcpyHostToDevice);
    tracePixel<<<(size+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(
            d_scene, d_raw, width, height, depth, samples);
    cudaDeviceSynchronize();
    cudaMemcpy(raw, d_raw, size * sizeof(glm::dvec3), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    // Draw
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            setPixel(x, y, raw[x + y * width]);
}

void GpuTracer::waitRender() {
    // There is nothing to wait for.
}