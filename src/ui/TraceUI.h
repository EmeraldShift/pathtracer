//
// rayUI.h
//
// The header file for the UI part
//

#ifndef __TraceUI_h__
#define __TraceUI_h__

#include <string>
#include <memory>

#define MAX_THREADS 32

using std::string;

class Tracer;

class CubeMap;

class TraceUI {
public:
    TraceUI() = default;

    virtual ~TraceUI() = default;

    virtual int run() = 0;

    virtual void setRayTracer(Tracer *r) { raytracer = r; }

    void useCubeMap(bool b) { m_usingCubeMap = b; }

    int getDepth() const { return m_nDepth; }

    int getSuperSamples() const { return m_nSuperSamples; }

    int getThreads() const { return m_threads; }

    CubeMap *getCubeMap() const { return cubemap.get(); }

    void setCubeMap(CubeMap *cm);

    static int m_threads; // number of threads to run
    static bool m_debug;
    bool m_gpu = false;
    int m_qoog = 64;

    static bool matchCubemapFiles(const string &one_cubemap_file,
                                  string matched_fn[6],
                                  string &pdir);

protected:
    Tracer *raytracer = nullptr;

    int m_nSize = 512;        // Size of the traced image
    int m_nDepth = 0;         // Max depth of recursion
    int m_nThreshold = 0;     // Threshold for interpolation within block
    int m_nBlockSize = 4;     // Blocksize (square, even, power of 2 preferred)
    int m_nSuperSamples = 2;  // Supersampling rate (1-d) for antialiasing
    int m_nAaThreshold = 100; // Pixel neighborhood difference for supersampling
    int m_nTreeDepth = 15;    // maximum kdTree depth
    int m_nLeafSize = 10;     // target number of objects per leaf
    int m_nFilterWidth = 1;   // width of cubemap filter

    // Determines whether or not to show debugging information
    // for individual rays.  Disabled by default for efficiency
    // reasons.
    bool m_displayDebuggingInfo = false;
    bool m_antiAlias = false;    // Is antialiasing on?
    bool m_kdTree = true;        // use kd-tree?
    bool m_shadows = true;       // compute shadows?
    bool m_smoothshade = true;   // turn on/off smoothshading?
    bool m_backface = true;      // cull backfaces?
    bool m_usingCubeMap = false; // render with cubemap
    bool m_internalReflection = true; // Enable reflection inside a translucent object.
    bool m_backfaceSpecular = false; // Enable specular component even seeing through the back of a translucent object.

    std::unique_ptr<CubeMap> cubemap;

    void loadFromJson(const char *file);

    void smartLoadCubemap(const string &file);
};

#endif
