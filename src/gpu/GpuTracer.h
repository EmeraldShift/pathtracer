#pragma once

#include "cuda.h"
#include "../tracer.h"

class GpuTracer : public Tracer {
public:
    GpuTracer(TraceUI *ui) : Tracer(ui) {}

    void traceImage(int width, int height) override;

    void waitRender() override;
};