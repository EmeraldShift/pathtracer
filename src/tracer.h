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
    }

    ~Tracer() = default;

    double aspectRatio();

    bool loadScene(const char *fn);

    void traceSetup(int w, int h);

    virtual void traceImage(int w, int h) = 0;

    virtual void waitRender() = 0;

    void getBuffer(unsigned char *&buf, int &w, int &h);

protected:
    void setPixel(int i, int j, f4 color);

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
    f4 tracePixel(int i, int j);

    f4 trace(float x, float y);

    f4 traceRay(ray &r, const f4 &thresh, int depth);

    unsigned threads;
    std::thread **workers;
};

class GpuTracer : public Tracer {
public:
    explicit GpuTracer(TraceUI *ui) : Tracer(ui) {
        threadsPerBlock = ui->m_qoog;
    }

    void traceImage(int width, int height) override;

    void waitRender() override;

private:
    unsigned threadsPerBlock;
};

#endif // __RAYTRACER_H__
