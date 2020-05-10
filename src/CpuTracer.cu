#include "tracer.h"

#include "scene/scene.h"

#include <atomic>
#include <mutex>
#include <random>
#include <iostream>
#include <glm/gtx/norm.hpp>

// For use in threading
struct WorkUnit {
    WorkUnit *next;
    int minX, maxX;
    int minY, maxY;
};
std::mutex l;
int totalWork;
WorkUnit *head = nullptr;
std::atomic<int> workDone;
std::atomic<int> threadsDone;

template<typename Numeric, typename Generator = std::mt19937>
static Numeric random(Numeric from, Numeric to) {
    thread_local static Generator gen(std::random_device{}());

    using dist_type = typename std::conditional<std::is_integral<Numeric>::value,
            std::uniform_int_distribution<Numeric>,
            std::uniform_real_distribution<Numeric>>::type;

    thread_local static dist_type dist;

    return dist(gen, typename dist_type::param_type{from, to});
}

static std::pair<glm::vec3, glm::vec3> getBasis(glm::vec3 normal) {
    auto a = std::abs(normal[0]) > RAY_EPSILON
             ? glm::normalize(glm::cross(glm::vec3(0, 1, 0), normal))
             : glm::normalize(glm::cross(glm::vec3(1, 0, 0), normal));
    auto b = glm::cross(normal, a);
    return std::make_pair(a, b);
}

static glm::vec3 randomVecFromHemisphere(glm::vec3 normal) {
    auto basis = getBasis(normal);
    auto p = 2 * 3.141592f * random<float>(0, 1);
    auto cos_p = std::cos(p);
    auto sin_p = std::sin(p);
    auto cos_t = std::pow(random<float>(0, 1), 2);
    auto sin_t = std::sqrt(1.0f - cos_t * cos_t);
    return sin_t * cos_p * basis.first + sin_t * sin_p * basis.second + cos_t * normal;
}

constexpr int granularity = 32;

void CpuTracer::traceImage(int w, int h) {
    for (int y = 0; y < h; y += granularity) {
        for (int x = 0; x < w; x += granularity) {
            auto *work = new WorkUnit;
            work->minX = x;
            work->minY = y;
            work->maxX = std::min(x + granularity, w);
            work->maxY = std::min(y + granularity, h);
            work->next = head;
            head = work;
            totalWork++;
        }
    }

    for (unsigned i = 0; i < threads; i++) {
        workers[i] = new std::thread([&, i] {
            while (true) {
                WorkUnit *work;
                l.lock();
                if (head) {
                    work = head;
                    head = head->next;
                    l.unlock();
                } else {
                    l.unlock();
                    break;
                }
                for (int y = work->minY; y < work->maxY; y++)
                    for (int x = work->minX; x < work->maxX; x++)
                        setPixel(x, y, tracePixel(x, y));
                delete work;

                // Announce progress
                int done;
                if ((done = ++workDone) % (std::max(1, totalWork / 20)) == 0)
                    std::cout << done * 100 / totalWork << "%" << std::endl;
            }
            threadsDone++;
        });
    }
}

glm::vec3 CpuTracer::tracePixel(int i, int j) {
    auto sum = glm::vec3();
    for (auto x = 0; x < samples; x++) {
        for (auto y = 0; y < samples; y++) {
            auto xx = float(i) - 0.5f + (1 + 2 * x) / (2.0f * samples);
            auto yy = float(j) - 0.5f + (1 + 2 * y) / (2.0f * samples);
            sum += trace(xx / float(buffer_width), yy / float(buffer_height));
        }
    }
    return sum / ((float) samples * samples);
}

glm::vec3 CpuTracer::trace(float x, float y) {
    ray r(glm::vec3(0, 0, 0), glm::vec3(0, 0, 0));
    scene->getCamera().rayThrough(x, y, r);
    glm::vec3 ret = traceRay(r, glm::vec3(1.0, 1.0, 1.0), depth);
    ret = glm::clamp(ret, 0.0f, 1.0f);
    return ret;
}

glm::vec3 CpuTracer::traceRay(ray &r, const glm::vec3 &thresh, int depth) {
    isect i;
    if (glm::length2(thresh) < 3.0 * 12.0 / 255.0 / 255.0)
        return glm::vec3(0);

    if (!scene->intersect(r, i))
        return cubeMap ? cubeMap->getColor(r) : glm::vec3();

    auto hitInner = r.getPosition() + (i.getT() + RAY_EPSILON) * r.getDirection();
    auto hitOuter = r.getPosition() + (i.getT() - RAY_EPSILON) * r.getDirection();

    if (depth < 0)
        return i.getMaterial().ke(i) * 32.0f;

    glm::vec3 rad(0);

    auto diffuse = glm::length(i.getMaterial().kd(i));
    auto reflect = i.getMaterial().kr(i).x;
    auto refract = i.getMaterial().kt(i).x;
    auto rand = random<float>(0, 1);

    auto refl = r.getDirection() - 2.0f * i.getN() * glm::dot(i.getN(), r.getDirection());
    if (rand < refract) {
        bool into = glm::dot(r.getDirection(), i.getN()) < 0;
        auto n = into ? i.getN() : -i.getN();
        auto n1 = 1.0f;
        auto n2 = i.getMaterial().index(i);
        auto ratio = into ? n1 / n2 : n2 / n1;
        auto dot = glm::dot(r.getDirection(), n);
        auto cos2t = 1 - ratio * ratio * (1 - dot * dot);
        if (cos2t < 0) {
            ray rr(hitOuter, refl);
            auto w = i.getMaterial().kr(i) / 1.0f;
            rad += w * traceRay(rr, w * thresh, depth - 1);
        } else {
            auto dir = glm::normalize(r.getDirection() * ratio - n * (dot * ratio + std::sqrt(cos2t)));
            ray rr(hitInner, dir);
            auto a = n2 - n1;
            auto b = n2 + n1;
            auto R0 = (a * a) / (b * b);
            auto c = 1.0f - (into ? -dot : glm::dot(rr.getDirection(), -n));
            auto Re = R0 + (1.0f - R0) * c * c * c * c * c;
            auto ratio2 = ratio * ratio;
            auto Tr = (1.0f - Re) * ratio2;

            auto prob = 0.25f + 0.5f * Re;
            // XXX depth test
            auto w = glm::vec3(1) * Tr / 1.0f;
            rad = w * traceRay(rr, w * thresh, depth - 1);
        }
    } else {
        auto basis = getBasis(i.getN());
        auto diff = randomVecFromHemisphere(i.getN());
        auto dir = glm::normalize(reflect * refl + (1 - reflect) * diff);
        ray rr(hitOuter, dir);
//        auto color = i.getMaterial().kd(i) + glm::vec3(0.15, 0.15, 0.15); // Makes even black a little reflective
        auto w = i.getMaterial().kd(i) / 1.0f;
        rad = w * (traceRay(rr, w * thresh, depth - 1));
//        auto w = glm::sqrt(color) / 1.0;
//        rad = w * traceRay(rr, w * thresh, depth - 1);
    }
//    else {
//        auto basis = getBasis(i.getN());
//        auto dir = randomVecFromHemisphere(i.getN());
//        ray rr(hitOuter, dir);
//        auto w = glm::vec3(1) / 1.0;
//        rad = w * (i.getMaterial().kd(i) + traceRay(rr, w * thresh, depth - 1)) / 2.0;
//    }
    return i.getMaterial().ke(i) * 32.0f + rad;
}

void CpuTracer::waitRender() {
    for (unsigned i = 0; i < threads; i++)
        workers[i]->join();
}