#pragma once

#include "material.h"
#include "transform.h"

class Trimesh;
class Geometry;

class TrimeshFace {
public:
    __host__ __device__ bool intersect(const ray &r, isect &i) const;

    static Geometry *create(const Material &mat, Trimesh *parent, int a, int b, int c);
private:
    TrimeshFace() = default;

    f4 vertices[3];
    f4 v0, v1, v2;
    f4 normal;
    f4 uInv, vInv, nInv;

    bool hasMaterials = false;
    Material materials[3];

    friend class Geometry;
};

class Trimesh {

public:
    explicit Trimesh(const Material &mat) : mat(mat) {}

    ~Trimesh();

    const Material &getMaterial() { return mat; }

    void setMaterial(const Material &m) { mat = m; }

    void addVertex(const f4 &v);

    void addMaterial(const Material &m);

    bool addFace(int a, int b, int c);

    std::vector<Geometry *> getFaces() { return faces; };

    const char *doubleCheck();

    static Trimesh *fromBox(Material &mat, TransformNode *transform);

    static Trimesh *fromSquare(Material &mat, TransformNode *transform);

    friend class TrimeshFace;

private:
    Trimesh *addVertices(f4 a, f4 b, f4 c);

    Trimesh *addCube(f4 *v);

    friend class TrimeshFace;

    Material mat;
    std::vector<f4> vertices;
    std::vector<Geometry *> faces;
    std::vector<Material> materials;
};