// The main ray tracer.

#pragma warning (disable: 4786)

#include "RayTracer.h"
#include "scene/light.h"
#include "scene/material.h"
#include "scene/ray.h"

#include "parser/Tokenizer.h"
#include "parser/Parser.h"

#include "ui/TraceUI.h"
#include <cmath>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtx/io.hpp>
#include <string.h> // for memset

#include <iostream>
#include <fstream>
#include <mutex>
#include <atomic>
#include <random>

using namespace std;
extern TraceUI *traceUI;

// Use this variable to decide if you want to print out
// debugging messages.  Gets set in the "trace single ray" mode
// in TraceGLWindow, for example.
bool debugMode = false;

// For use in threading
struct WorkUnit {
    WorkUnit *next;
    int minX, maxX;
    int minY, maxY;
};
std::mutex l;
WorkUnit *head = nullptr;
std::atomic<int> done;

constexpr int DEPTH_LIMIT = 4;
constexpr int SUPER_MAX_DEPTH = 8;
constexpr int SUPERSAMPLE = 10;
constexpr int SAMPLES = 2;

// Trace a top-level ray through pixel(i,j), i.e. normalized window coordinates (x,y),
// through the projection plane, and out into the scene.  All we do is
// enter the main ray-tracing method, getting things started by plugging
// in an initial ray weight of (0.0,0.0,0.0) and an initial recursion depth of 0.

template<typename Numeric, typename Generator = std::mt19937>
Numeric random(Numeric from, Numeric to) {
    thread_local static Generator gen(std::random_device{}());

    using dist_type = typename std::conditional<std::is_integral<Numeric>::value,
            std::uniform_int_distribution<Numeric>,
            std::uniform_real_distribution<Numeric>>::type;

    thread_local static dist_type dist;

    return dist(gen, typename dist_type::param_type{from, to});
}

static std::pair<glm::dvec3, glm::dvec3> getBasis(glm::dvec3 normal) {
    auto a = normal[0] > RAY_EPSILON ? glm::normalize(glm::cross(glm::dvec3(0, 1, 0), normal))
                                     : glm::normalize(glm::cross(glm::dvec3(1, 0, 0), normal));
    auto b = glm::cross(normal, a);
    return std::make_pair(a, b);
}

static glm::dvec3 randomVecFromHemisphere(glm::dvec3 normal) {
    auto basis = getBasis(normal);
    auto p = 2 * M_PI * random<double>(0, 1);
    auto cos_p = std::cos(p);
    auto sin_p = std::sin(p);
    auto cos_t = std::pow(random<double>(0, 1), 2);
    auto sin_t = std::sqrt(1.0 - cos_t * cos_t);
    return sin_t * cos_p * basis.first + sin_t * sin_p * basis.second + cos_t * normal;
}

glm::dvec3 RayTracer::trace(double x, double y) {
    // Clear out the ray cache in the scene for debugging purposes,
    if (TraceUI::m_debug)
        scene->intersectCache.clear();

    ray r(glm::dvec3(0, 0, 0), glm::dvec3(0, 0, 0), glm::dvec3(1, 1, 1), ray::VISIBILITY);
    scene->getCamera().rayThrough(x, y, r);
    double dummy;
    glm::dvec3 ret = traceRay(r, glm::dvec3(1.0, 1.0, 1.0), traceUI->getDepth(), dummy);
    ret = glm::clamp(ret, 0.0, 1.0);
    return ret;
}

glm::dvec3 RayTracer::tracePixel(int i, int j) {
    glm::dvec3 col(0, 0, 0);

    if (!sceneLoaded()) return col;

    double x = double(i) / double(buffer_width);
    double y = double(j) / double(buffer_height);

    unsigned char *pixel = buffer.data() + (i + j * buffer_width) * 3;
    col = trace(x, y);

    pixel[0] = (int) (255.0 * col[0]);
    pixel[1] = (int) (255.0 * col[1]);
    pixel[2] = (int) (255.0 * col[2]);
    return col;
}

#define VERBOSE 0

// Do recursive ray tracing!  You'll want to insert a lot of code here
// (or places called from here) to handle reflection, refraction, etc etc.
glm::dvec3 RayTracer::traceRay(ray &r, const glm::dvec3 &thresh, int depth, double &t) {
    isect i;
    glm::dvec3 colorC;

    if (!scene->intersect(r, i))
        return traceUI->cubeMap() ? traceUI->getCubeMap()->getColor(r) : glm::dvec3();

    auto hitInner = r.getPosition() + (i.getT() - RAY_EPSILON) * r.getDirection();
    auto hitOuter = r.getPosition() + (i.getT() + RAY_EPSILON) * r.getDirection();

    double rr_prob = std::max({i.getMaterial().kd(i)[0], i.getMaterial().kd(i)[1], i.getMaterial().kd(i)[2]});
    if (depth < -DEPTH_LIMIT)
        rr_prob *= std::pow(0.5, -DEPTH_LIMIT - depth);
    if (depth < 0) {
        if (random<double>(0, 1) > rr_prob)
            return i.getMaterial().ke(i);
    } else {
        rr_prob = 1;
    }

    glm::dvec3 rad, w;

    if (glm::length2(i.getMaterial().kd(i)) > RAY_EPSILON) {
        auto basis = getBasis(i.getN());
        auto refl = randomVecFromHemisphere(i.getN());
        ray rr(hitOuter, refl, glm::dvec3());
        rad += (i.getMaterial().kd(i) * traceRay(rr, thresh, depth - 1, t)) / rr_prob;
    }

    return i.getMaterial().ke(i) * 32.0 + rad;
}

RayTracer::RayTracer()
        : scene(nullptr), buffer(0), thresh(0), buffer_width(0), buffer_height(0), m_bBufferReady(false) {
}

RayTracer::~RayTracer() {
}

void RayTracer::getBuffer(unsigned char *&buf, int &w, int &h) {
    buf = buffer.data();
    w = buffer_width;
    h = buffer_height;
}

double RayTracer::aspectRatio() {
    return sceneLoaded() ? scene->getCamera().getAspectRatio() : 1;
}

bool RayTracer::loadScene(const char *fn) {
    ifstream ifs(fn);
    if (!ifs) {
        string msg("Error: couldn't read scene file ");
        msg.append(fn);
        traceUI->alert(msg);
        return false;
    }

    // Strip off filename, leaving only the path:
    string path(fn);
    if (path.find_last_of("\\/") == string::npos)
        path = ".";
    else
        path = path.substr(0, path.find_last_of("\\/"));

    // Call this with 'true' for debug output from the tokenizer
    Tokenizer tokenizer(ifs, false);
    Parser parser(tokenizer, path);
    try {
        scene.reset(parser.parseScene());
    }
    catch (SyntaxErrorException &pe) {
        traceUI->alert(pe.formattedMessage());
        return false;
    } catch (ParserException &pe) {
        string msg("Parser: fatal exception ");
        msg.append(pe.message());
        traceUI->alert(msg);
        return false;
    } catch (TextureMapException e) {
        string msg("Texture mapping exception: ");
        msg.append(e.message());
        traceUI->alert(msg);
        return false;
    }

    if (!sceneLoaded())
        return false;

    scene->constructBvh();

    return true;
}

void RayTracer::traceSetup(int w, int h) {
    size_t newBufferSize = w * h * 3;
    if (newBufferSize != buffer.size()) {
        bufferSize = newBufferSize;
        buffer.resize(bufferSize);
    }
    buffer_width = w;
    buffer_height = h;
    std::fill(buffer.begin(), buffer.end(), 0);
    m_bBufferReady = true;

    /*
     * Sync with TraceUI
     */

    threads = traceUI->getThreads();
    block_size = traceUI->getBlockSize();
    thresh = traceUI->getThreshold();
    samples = traceUI->getSuperSamples();
    aaThresh = traceUI->getAaThreshold();
    done = 0;
    workers = new std::thread*[threads];
}

constexpr int granularity = 32;

/*
 * RayTracer::traceImage
 *
 *	Trace the image and store the pixel data in RayTracer::buffer.
 *
 *	Arguments:
 *		w:	width of the image buffer
 *		h:	height of the image buffer
 *
 */
void RayTracer::traceImage(int w, int h) {
    traceSetup(w, h);

    for (int y = 0; y < h; y += granularity) {
        for (int x = 0; x < w; x += granularity) {
            auto *work = new WorkUnit;
            work->minX = x;
            work->minY = y;
            work->maxX = std::min(x + granularity, w);
            work->maxY = std::min(y + granularity, h);
            work->next = head;
            head = work;
        }
    }

    for (int i = 0; i < threads; i++) {
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
                        tracePixel(x, y);
                delete work;
            }
            done++;
        });
    }
}

bool RayTracer::checkRender() {
    return done == threads;
}

void RayTracer::waitRender() {
    for (int i = 0; i < threads; i++)
        workers[i]->join();
}


glm::dvec3 RayTracer::getPixel(int i, int j) {
    unsigned char *pixel = buffer.data() + (i + j * buffer_width) * 3;
    return glm::dvec3((double) pixel[0] / 255.0, (double) pixel[1] / 255.0, (double) pixel[2] / 255.0);
}

void RayTracer::setPixel(int i, int j, glm::dvec3 color) {
    unsigned char *pixel = buffer.data() + (i + j * buffer_width) * 3;

    pixel[0] = (int) (255.0 * color[0]);
    pixel[1] = (int) (255.0 * color[1]);
    pixel[2] = (int) (255.0 * color[2]);
}

