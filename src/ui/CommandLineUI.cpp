#include <cstdarg>
#include <ctime>
#include <iostream>
#ifndef _MSC_VER
#include <unistd.h>
#else
extern char* optarg;
extern int optind, opterr, optopt;
extern int getopt(int argc, char** argv, const char* optstring);
#endif

#include <cassert>

#include "../fileio/images.h"
#include "CommandLineUI.h"

#include "../tracer.h"

// The command line UI simply parses out all the arguments off
// the command line and stores them locally.
CommandLineUI::CommandLineUI(int argc, char** argv) : TraceUI()
{
	int i;
	progName = argv[0];
	const char* jsonfile = nullptr;
	string cubemap_file;
	while ((i = getopt(argc, argv, "t:r:w:hj:c:s:g")) != EOF) {
		switch (i) {
		    case 't':
		        m_threads = atoi(optarg);
			case 'r':
				m_nDepth = atoi(optarg);
				break;
			case 'w':
				m_nSize = atoi(optarg);
				break;
			case 'j':
				jsonfile = optarg;
				break;
			case 'c':
				cubemap_file = optarg;
				break;
		    case 's':
		        m_nSuperSamples = atoi(optarg);
		        break;
		    case 'g':
		        m_gpu = true;
		        break;
			case 'h':
				usage();
				exit(1);
			default:
				// Oops; unknown argument
				std::cerr << "Invalid argument: '" << i << "'."
				          << std::endl;
				usage();
				exit(1);
		}
	}
	if (jsonfile) {
		loadFromJson(jsonfile);
	}
	if (!cubemap_file.empty()) {
		smartLoadCubemap(cubemap_file);
	}

	if (optind >= argc - 1) {
		std::cerr << "no input and/or output name." << std::endl;
		exit(1);
	}

	rayName = argv[optind];
	imgName = argv[optind + 1];
}

int CommandLineUI::run()
{
	assert(raytracer != nullptr);
	if (raytracer->loadScene(rayName)) {
		int width = m_nSize;
		int height = (int)std::round(width / raytracer->aspectRatio() + 0.5);

		raytracer->traceSetup(width, height);
		raytracer->traceImage(width, height);
		raytracer->waitRender();

		// save image
		unsigned char* buf;
		raytracer->getBuffer(buf, width, height);
		if (buf)
			writeImage(imgName, width, height, buf);
		return 0;
	} else {
		std::cerr << "Unable to load ray file '" << rayName << "'"
		          << std::endl;
		return 1;
	}
}

void CommandLineUI::usage()
{
	std::cerr << "usage: " << progName
	     << " [options] [input.ray output.png]" << std::endl
	     << "  -r <#>      set recursion level (default " << m_nDepth << ")" << std::endl
	     << "  -w <#>      set output image width (default " << m_nSize << ")" << std::endl
	     << "  -j <FILE>   set parameters from JSON file" << std::endl
	     << "  -c <FILE>   one Cubemap file, the remainings will be detected automatically" << std::endl;
}
