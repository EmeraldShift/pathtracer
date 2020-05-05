// #ifndef COMMAND_LINE_ONLY
// #include "ui/GraphicalUI.h"
// #endif

#include "tracer.h"
#include "ui/CommandLineUI.h"
#include "gpu/cuda.h"

using namespace std;

int TraceUI::m_threads = max(std::thread::hardware_concurrency(), (unsigned)1);
int TraceUI::rayCount[MAX_THREADS];

int main(int argc, char** argv)
{
    TraceUI *ui = new CommandLineUI(argc, argv);
    ui->setRayTracer(new CpuTracer(ui));
	return ui->run();
}
