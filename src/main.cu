#include "tracer.h"
#include "ui/CommandLineUI.h"
#include "scene/bvh.h"

int TraceUI::m_threads = std::max(std::thread::hardware_concurrency(), (unsigned) 1);

int main(int argc, char **argv) {
    TraceUI *ui = new CommandLineUI(argc, argv);
    ui->m_gpu ? ui->setRayTracer(new GpuTracer(ui))
              : ui->setRayTracer(new CpuTracer(ui));
    return ui->run();

}
