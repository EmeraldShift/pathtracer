#include "tracer.h"

#include "scene/geometry.h"
#include "scene/scene.h"

#include <curand_kernel.h>
#include "cuda_profiler_api.h"

#define abs(a) (a < 0 ? -a : a)

struct Pair {
    f4 a, b;
};

__device__ static Pair
getBasis(f4 normal) {
    auto a = abs(normal[0]) > RAY_EPSILON
             ? f4m::normalize(f4m::cross({0, 1, 0}, normal))
             : f4m::normalize(f4m::cross({1, 0, 0}, normal));
    auto b = f4m::cross(normal, a);
    return {a, b};
}

__device__ static f4
randomVecFromHemisphere(f4 normal, curandState &state) {
    auto basis = getBasis(normal);
    float p = 2 * 3.141592f * curand_uniform(&state);
    float cos_p = cos(p);
    float sin_p = sin(p);
    float cos_t = pow(curand_uniform(&state), 2);
    float sin_t = sqrt(1.0f - cos_t * cos_t);
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

 //returns true if successfully generated a new ray
 __device__ static bool
 genRay(ray& r, f4& thresh, f4& color, int& depth, int og_depth, int& index, curandState &state, ){
    //check State at current index (read)
    //if count < sample* sample (check atomic add return) && x, y != -1
    //atomic add to count
    //sx = count / samples;
    //sy = count % samples;
    //gen ray, restart thingies
    //else
    //set visited on current
    //loop over rest until find one with bit not set, then do the above
    //if you hit the one = threadIdx, then return false
    auto xx = float(x) - 0.5f + (1 + 2 * sx) / (2.0f * samples);
    auto yy = float(y) - 0.5f + (1 + 2 * sy) / (2.0f * samples);
    scene->getCamera().rayThrough(xx / float(width), yy / float(height), r);
 }
__device__ static f4
traceRay(Scene *scene, ray &r, int depth, curandState &state) {
    f4 color;
    f4 thresh(1);
    int og_depth = depth;
    int pixel = threadIdx.x;
    
    while (true) {
        isect i;
        if (f4m::length2(thresh) < 3.0f * 12.0f / 255.0f / 255.0f){
            //genray
        }

        // Stop tracing if we exceed depth limit
        if (depth < 0) {
            color *= thresh * i.getMaterial().ke(i) * 32.0f;
            color = f4m::clamp(color, 0.0, 1.0);
            cache[pixel] += color; //ATOOMIC ADD
            //genray
        }
        // Stop tracing if we miss the scene
        if (!scene->intersectIterative(r, i)) {
            color += thresh;
            color = f4m::clamp(color, 0.0, 1.0);
            cache[pixel] += color; //ATOOMIC ADD
            //genray
            continue;
        }

        

        auto hitInner = r.getPosition() + (i.getT() + RAY_EPSILON) * r.getDirection();
        auto hitOuter = r.getPosition() + (i.getT() - RAY_EPSILON) * r.getDirection();


        auto diffuse = f4m::length(i.getMaterial().kd(i));
        auto reflect = i.getMaterial().kr(i)[0];
        auto refract = i.getMaterial().kt(i)[0];
        auto rand = curand_uniform_double(&state);

        // New ray generation, either refractive or diffuse
        auto refl = r.getDirection() - 2.0f * i.getN() * f4m::dot(i.getN(), r.getDirection());
        if (rand < refract) {
            bool into = f4m::dot(r.getDirection(), i.getN()) < 0;
            auto n = into ? i.getN() : -i.getN();
            auto n1 = 1.0f;
            auto n2 = i.getMaterial().index(i);
            auto ratio = into ? n1 / n2 : n2 / n1;
            auto dot = f4m::dot(r.getDirection(), n);
            auto cos2t = 1 - ratio * ratio * (1 - dot * dot);
            if (cos2t < 0) {
                r = ray(hitOuter, refl);
                color += thresh * i.getMaterial().ke(i) * 32.0f;
                thresh *= i.getMaterial().kr(i) / 1.0f;
                depth--;
            } else {
                auto dir = f4m::normalize(r.getDirection() * ratio - n * (dot * ratio + sqrt(cos2t)));
                auto a = n2 - n1;
                auto b = n2 + n1;
                auto R0 = (a * a) / (b * b);
                auto c = 1.0f - (into ? -dot : f4m::dot(dir, -n));
                auto Re = R0 + (1.0f - R0) * c * c * c * c * c;
                auto ratio2 = ratio * ratio;
                auto Tr = (1.0f - Re) * ratio2;

                auto prob = 0.25f + 0.5f * Re;
                // XXX depth test
                r = ray(hitInner, dir);
                color += thresh * i.getMaterial().ke(i) * 32.0f;
                thresh *= Tr;
                depth--;
            }
        } else {
            auto diff = randomVecFromHemisphere(i.getN(), state);
            auto dir = f4m::normalize(reflect * refl + (1 - reflect) * diff);
            r = ray(hitOuter, dir);
            color += thresh * i.getMaterial().ke(i) * 32.0f;
            thresh *= i.getMaterial().kd(i);
            depth--;
        }
    }
    return color;
}
struct State{
    int count; //count of samples already initiated
    int x; //x, y of pixel being generated
    int y;
    f4 sum;
    bool done; //if looped around an all progress, then terminate
}
__global__ static void
tracePixel(Scene *scene, f4 *buffer, unsigned width, unsigned height, unsigned depth, unsigned samples) {
    //unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
    // unsigned x = idx % width; //threadID controls x
    // unsigned y = idx / width; //blockId controls y

    __shared__ State cache[64];

    //alt
    int row_in = threadIdx.x / 8;
    int col_in  = threadIdx.x % 8;
    int row_out = blockIdx.x /64;
    int col_out = blockIdx.x % 64;

    int x = col_in + col_out*8;
    int y = row_in + row_out*8;

    unsigned idx = x + y*width;


    if (idx < width * height) {
        cache[threadIdx.x ] = State(0, x, y, f4(0));
    } else{
        cache[threadId.x] = State(0, -1, -1, f4(0));
    }
        curandState state;
        curand_init((unsigned long long) clock() + idx, 0, 0, &state);
        f4 sum;

        // Trace a ray for each subpixel, add up, and normalize.
        for (auto sx = 0; sx < samples; sx++) {
            for (auto sy = 0; sy < samples; sy++) {

            }
        }
    }

    if (idx < width * height)
        buffer[idx] = cache[threadIdx.x].sum/(samples*samples);
}

void GpuTracer::traceImage(int width, int height) {
    cudaProfilerStart();
    std::cout<<width<<" "<<height<<std::endl;
    // Allocate a buffer of color vectors per pixel
    auto size = width * height;
    auto raw = new f4[size];
    f4 *d_raw;
    cudaMalloc((void **) &d_raw, size * sizeof(f4));

    // Does a lot of work--copies the entire scene and returns
    // a pointer to the scene in device memory.
    // See: Scene::clone, BoundedVolumeHierarchy::clone,
    // Cluster::clone, and Geometry(& Sphere/Trimesh)::clone.
    auto d_scene = scene->clone();

    // Yeet the pixel buffer to the device, render, and fetch the values.
    cudaMemcpy(d_raw, raw, size * sizeof(f4), cudaMemcpyHostToDevice);
    tracePixel<<<(size+threadsPerBlock-1)/threadsPerBlock, threadsPerBlock>>>(
            d_scene, d_raw, width, height, depth, samples);
    cudaDeviceSynchronize();
    cudaMemcpy(raw, d_raw, size * sizeof(f4), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaProfilerStop();

    // Draw
    for (int y = 0; y < height; y++)
        for (int x = 0; x < width; x++)
            setPixel(x, y, raw[x + y * width]);
}

void GpuTracer::waitRender() {
    // There is nothing to wait for.
}