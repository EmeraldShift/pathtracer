#include "tracer.h"
#include "ui/CommandLineUI.h"
#include "gpu/GpuTracer.h"

int TraceUI::m_threads = std::max(std::thread::hardware_concurrency(), (unsigned)1);

int main(int argc, char** argv)
{
    TraceUI *ui = new CommandLineUI(argc, argv);
    Tracer *tracer;
    if (ui->m_gpu)
        tracer = new GpuTracer(ui);
    else
        tracer = new CpuTracer(ui);
    ui->setRayTracer(tracer);
	return ui->run();
}
