#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"
#include "../gl.h"
#include <glm/vec3.hpp>
#include <glm/mat3x3.hpp>

class Camera
{
public:
    CUDA_CALLABLE_MEMBER Camera() = default;
    CUDA_CALLABLE_MEMBER void rayThrough( double x, double y, ray &r ) const;
    CUDA_CALLABLE_MEMBER void setEye( const glm::dvec3 &eye );
    CUDA_CALLABLE_MEMBER void setLook( double, double, double, double );
    CUDA_CALLABLE_MEMBER void setLook( const glm::dvec3 &viewDir, const glm::dvec3 &upDir );
    CUDA_CALLABLE_MEMBER void setFOV( double );
    CUDA_CALLABLE_MEMBER void setAspectRatio( double );
    CUDA_CALLABLE_MEMBER double getAspectRatio() const { return aspectRatio; }

private:
    CUDA_CALLABLE_MEMBER void update();

    double normalizedHeight = 1;
    double aspectRatio = 1;
    glm::dmat3 m = glm::dmat3x3(1);
    glm::dvec3 eye = glm::dvec3(0, 0, 0);
    glm::dvec3 u = glm::dvec3(1,0, 0);
    glm::dvec3 v = glm::dvec3(0, 1, 0);
    glm::dvec3 look = glm::dvec3(0, 1, -1);
};

#endif
