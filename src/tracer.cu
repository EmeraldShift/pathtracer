// The main ray tracer.

#pragma warning (disable: 4786)

#include "tracer.h"

#include "parser/Tokenizer.h"
#include "parser/Parser.h"

#include "ui/TraceUI.h"
#include <cmath>
#include <algorithm>
#include <glm/gtx/io.hpp>
#include <cstring> // for memset

#include <iostream>
#include <fstream>
#include <random>

constexpr double W = 2.0;
constexpr double exposureBias = 2.0;
static double hable(double x) {
    double a = 0.15;
    double b = 0.50;
    double c = 0.10;
    double d = 0.20;
    double e = 0.02;
    double f = 0.30;
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

void Tracer::getBuffer(unsigned char *&buf, int &w, int &h) {
    buf = buffer.data();
    w = buffer_width;
    h = buffer_height;
}

double Tracer::aspectRatio() {
    return scene->getCamera().getAspectRatio();
}

bool Tracer::loadScene(const char *fn) {
    std::ifstream ifs(fn);
    if (!ifs) {
        std::string msg("Error: couldn't read scene file ");
        msg.append(fn);
        std::cerr << msg << std::endl;
        return false;
    }

    // Strip off filename, leaving only the path:
    std::string path(fn);
    if (path.find_last_of("\\/") == std::string::npos)
        path = ".";
    else
        path = path.substr(0, path.find_last_of("\\/"));

    // Call this with 'true' for debug output from the tokenizer
    Tokenizer tokenizer(ifs, false);
    Parser parser(tokenizer, path);
    try {
        scene = parser.parseScene();
    }
    catch (SyntaxErrorException &pe) {
        std::cerr << pe.formattedMessage() << std::endl;
        return false;
    } catch (ParserException &pe) {
        std::string msg("Parser: fatal exception ");
        msg.append(pe.message());
        std::cerr << msg << std::endl;
        return false;
    } catch (TextureMapException e) {
        std::string msg("Texture mapping exception: ");
        msg.append(e.message());
        std::cerr << msg << std::endl;
        return false;
    }

    //ALL OBJECTS HAVE BEEN LOADED, UPDATE obj to contain pointers to actual
    scene->updateObjects();

    if (scene)
        scene->constructBvh();
    return scene != nullptr;
}

void Tracer::traceSetup(int w, int h) {
    auto size = w * h * 3;
    buffer.resize(size);
    buffer_width = w;
    buffer_height = h;
    std::fill(buffer.begin(), buffer.end(), 0);
}

void Tracer::setPixel(int i, int j, f4 color) {
    unsigned char *pixel = buffer.data() + (i + j * buffer_width) * 3;
    pixel[0] = (int) (255.0 * (1.0 / hable(W)) * hable(exposureBias * color[0]));
    pixel[1] = (int) (255.0 * (1.0 / hable(W)) * hable(exposureBias * color[1]));
    pixel[2] = (int) (255.0 * (1.0 / hable(W)) * hable(exposureBias * color[2]));
}