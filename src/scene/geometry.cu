#include "geometry.h"

Geometry *Geometry::clone() const {
    Geometry *d_geom;
    cudaMalloc(&d_geom, sizeof(Geometry));
    cudaMemcpy(d_geom, this, sizeof(Geometry), cudaMemcpyHostToDevice);
    return d_geom;
}