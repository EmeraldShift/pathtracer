#ifndef __RAYTRACER_H__
#define __RAYTRACER_H__

#define MAX_THREADS 32

// The main ray tracer.

#include <time.h>
#include <glm/vec3.hpp>
#include <queue>
#include <thread>
#include <iostream>
#include "scene/cubeMap.h"
#include "scene/ray.h"
#include "ui/TraceUI.h"

class Scene;

class Tracer {
public:
    Tracer(TraceUI *traceUi) :
            depth(traceUi->getDepth()),
            samples(traceUi->getSuperSamples()),
            cubeMap(traceUi->getCubeMap()) {
        std::cout << "tracer got " << cubeMap << std::endl;
    }

    ~Tracer() = default;

    double aspectRatio();

    bool loadScene(const char *fn);

    void traceSetup(int w, int h);

    virtual void traceImage(int w, int h) = 0;

    virtual void waitRender() = 0;

    void getBuffer(unsigned char *&buf, int &w, int &h);

protected:
    void setPixel(int i, int j, glm::dvec3 color);

    unsigned depth;
    unsigned samples;
    CubeMap *cubeMap;

    Scene *scene;

    std::vector<unsigned char> buffer;
    int buffer_width, buffer_height;
};

class CpuTracer : public Tracer {
public:
    CpuTracer(TraceUI *traceUi)
            : Tracer(traceUi),
              threads(traceUi->getThreads()),
              workers(new std::thread *[threads]) {}

    ~CpuTracer() = default;

    void traceImage(int w, int h) override;

    void waitRender() override;

private:
    glm::vec3 tracePixel(int i, int j);

    glm::vec3 trace(float x, float y);

    glm::vec3 traceRay(ray &r, const glm::vec3 &thresh, int depth);

    unsigned threads;
    std::thread **workers;
};

class GpuTracer : public Tracer {
public:
    explicit GpuTracer(TraceUI *ui) : Tracer(ui) {}

    void traceImage(int width, int height) override;

    void waitRender() override;
};

#endif // __RAYTRACER_H__
