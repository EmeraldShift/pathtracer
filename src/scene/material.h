//
// material.h
//
// The Material class: a description of the phsyical properties of a surface
// that are used to determine how that surface interacts with light.

#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <utility>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include "../vec.h"

class ray;

class isect;

using std::string;

/*
 * The TextureMap class can be used to store a texture map,
 * which consists of a bitmap and various accessors to
 * it.  To implement basic texture mapping, you'll want to
 * fill in the getMappedValue function to implement basic
 * texture mapping.
 */
class TextureMap {
public:
    explicit TextureMap(const string &filename);

    // Return the mapped value; here the coordinate
    // is assumed to be within the parametrization space:
    // [0, 1] x [0, 1]
    // (i.e., {(u, v): 0 <= u <= 1 and 0 <= v <= 1}
    f4 getMappedValue(const float2 &coord) const;

    // Retrieve the value stored in a physical location
    // (with integer coordinates) in the bitmap.
    // Should be called from getMappedValue in order to
    // do bilinear interpolation.
    f4 getPixelAt(int x, int y) const;

    ~TextureMap() = default;

protected:
    int width;
    int height;
    std::vector<uint8_t> data;
};

class TextureMapException {
public:
    explicit TextureMapException(string errorMsg) : _errorMsg(std::move(errorMsg)) {}

    std::string message() { return _errorMsg; }

private:
    std::string _errorMsg;
};

/*
 * MaterialParameter is a helper class for a material;
 * it stores either a constant value (in a 3-vector)
 * or else a link to a map of some type.  If the pointer
 * to the TextureMap is NULL, the class simply returns
 * whatever is stored in the constant vector.
 *
 * This is unabashedly a speed hack; we are replacing a
 * potential use of polymorphism with an old-school "if"
 * statement.  But raytracers are slow enough, and the
 * Material class is used often enough, that it is
 * (somewhat) justifiable to do this.
 */

class Material {

public:
    Material() = default;

    ~Material() = default;

    Material(const f4 &e, const f4 &d,
             const f4 &r, const f4 &t, float in)
            : _ke(e), _kd(d), _kr(r), _kt(t), _index(glm::vec3(in, in, in)) {}

    __host__ __device__
    Material &
    operator+=(const Material &m) {
        _ke += m._ke;
        _kd += m._kd;
        _kr += m._kr;
        _kt += m._kt;
        _index += m._index;
        return *this;
    }

    __host__ __device__ friend Material operator*(float d, Material m);

    // Accessor functions; we pass in an isect& for cases where
    // the parameter is dependent on, for example, world-space
    // coordinates (i.e., solid textures) or parametrized coordinates
    // (i.e., mapped textures)
    __host__ __device__ f4 ke(const isect &i) const { return _ke; }

    __host__ __device__ f4 kd(const isect &i) const { return _kd; }

    __host__ __device__ f4 kr(const isect &i) const { return _kr; }

    __host__ __device__ f4 kt(const isect &i) const { return _kt; }

    __host__ __device__ float index(const isect &i) const { return _index[0]; }

    void setEmissive(const f4 &ke) { _ke = ke; }

    void setDiffuse(const f4 &kd) { _kd = kd; }

    void setReflective(const f4 &kr) { _kr = kr; }

    void setTransmissive(const f4 &kt) { _kt = kt; }

    void setIndex(const f4 &index) { _index = index; }

private:
    f4 _ke;
    f4 _kd;
    f4 _kr;
    f4 _kt;
    f4 _index = 1;
};

// This doesn't necessarily make sense for mapped materials
__host__ __device__
inline Material
operator*(float d, Material m) {
    m._ke *= d;
    m._kd *= d;
    m._kr *= d;
    m._kt *= d;
    m._index *= d;
    return m;
}

#endif // __MATERIAL_H__
