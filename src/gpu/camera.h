#pragma once

class GpuCamera {
public:
    void rayThrough( double x, double y, ray &r ) const;

private:
    double normalizedHeight;
    double aspectRatio;
    glm::dmat3 m;
    glm::dvec3 eye;
    glm::dvec3 u;
    glm::dvec3 ;
    glm::dvec3 look;
};