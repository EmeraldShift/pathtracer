//
// material.h
//
// The Material class: a description of the phsyical properties of a surface
// that are used to determine how that surface interacts with light.

#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include "../gpu/cuda.h"

#include <utility>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <cstdint>

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
    explicit TextureMap(const string& filename);

    // Return the mapped value; here the coordinate
    // is assumed to be within the parametrization space:
    // [0, 1] x [0, 1]
    // (i.e., {(u, v): 0 <= u <= 1 and 0 <= v <= 1}
    glm::dvec3 getMappedValue(const glm::dvec2 &coord) const;

    // Retrieve the value stored in a physical location
    // (with integer coordinates) in the bitmap.
    // Should be called from getMappedValue in order to
    // do bilinear interpolation.
    glm::dvec3 getPixelAt(int x, int y) const;

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

class MaterialParameter {
public:
    explicit MaterialParameter(const glm::dvec3 &par)
            : _value(par) {
    }

    explicit MaterialParameter(const double par)
            : _value(par, par, par) {
    }

    explicit MaterialParameter(TextureMap *tex)
            : _textureMap(tex) {
    }

    MaterialParameter() = default;

    MaterialParameter &operator*=(const MaterialParameter &rhs) {
        (*this) *= rhs._value;
        return *this;
    }

    glm::dvec3 &operator*=(const glm::dvec3 &rhs) {
        _value += rhs;
//        _value[0] *= rhs[0];
//        _value[1] *= rhs[1];
//        _value[2] *= rhs[2];
        return _value;
    }

    glm::dvec3 &operator*=(const double rhs) {
        _value[0] *= rhs;
        _value[1] *= rhs;
        _value[2] *= rhs;
        return _value;
    }

    MaterialParameter &operator+=(const MaterialParameter &rhs) {
        _value += rhs._value;
        return *this;
    }

    glm::dvec3 &operator+=(const glm::dvec3 &rhs) {
        _value += rhs;
        return _value;
    }

    glm::dvec3 value(const isect &is) const;

    double intensityValue(const isect &is) const;

    // Use this to determine if the particular parameter is
    // mapped; use this to determine if we need to somehow renormalize.
    bool mapped() const { return _textureMap != nullptr; }

private:
    glm::dvec3 _value = glm::dvec3(0);
    TextureMap *_textureMap = nullptr;
};

class Material {

public:
    Material() = default;

    virtual ~Material() = default;

    Material(const glm::dvec3 &e, const glm::dvec3 &d,
             const glm::dvec3 &r, const glm::dvec3 &t, double in)
            : _ke(e), _kd(d), _kr(r), _kt(t), _index(glm::dvec3(in, in, in)) {}

    Material &
    operator+=(const Material &m) {
        _ke += m._ke;
        _kd += m._kd;
        _kr += m._kr;
        _kt += m._kt;
        _index += m._index;
        return *this;
    }

    friend Material operator*(double d, Material m);

    // Accessor functions; we pass in an isect& for cases where
    // the parameter is dependent on, for example, world-space
    // coordinates (i.e., solid textures) or parametrized coordinates
    // (i.e., mapped textures)
    glm::dvec3 ke(const isect &i) const { return _ke.value(i); }

    glm::dvec3 kd(const isect &i) const { return _kd.value(i); }

    glm::dvec3 kr(const isect &i) const { return _kr.value(i); }

    glm::dvec3 kt(const isect &i) const { return _kt.value(i); }

    double index(const isect &i) const { return _index.intensityValue(i); }

    // setting functions taking MaterialParameters
    void setEmissive(const MaterialParameter &ke) { _ke = ke; }

    void setDiffuse(const MaterialParameter &kd) { _kd = kd; }

    void setReflective(const MaterialParameter &kr) {
        _kr = kr;
    }

    void setTransmissive(const MaterialParameter &kt) {
        _kt = kt;
    }

    void setIndex(const MaterialParameter &index) { _index = index; }

private:
    MaterialParameter _ke = MaterialParameter(glm::dvec3(0));
    MaterialParameter _kd = MaterialParameter(glm::dvec3(0));
    MaterialParameter _kr = MaterialParameter(glm::dvec3(0));
    MaterialParameter _kt = MaterialParameter(glm::dvec3(0));
    MaterialParameter _index = MaterialParameter(1.0);
};

// This doesn't necessarily make sense for mapped materials
inline Material
operator*(double d, Material m) {
    m._ke *= d;
    m._kd *= d;
    m._kr *= d;
    m._kt *= d;
    m._index *= d;
    return m;
}

#endif // __MATERIAL_H__
