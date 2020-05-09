#pragma once

#include "geometry.h"
#include "material.h"
#include "transform.h"

class Trimesh;

class TrimeshFace : public Geometry {
public:
    glm::dvec3 getNormal() { return normal; }

    CUDA_CALLABLE_MEMBER static bool intersect(void *obj, ray &r, isect &i);

    TrimeshFace *clone() const override;

    static TrimeshFace *create(const Material &mat, Trimesh *parent, int a, int b, int c);

private:
    TrimeshFace() = default;

    BoundingBox computeLocalBoundingBox();

    glm::dvec3 vertices[3];
    glm::dvec3 v0 = glm::dvec3();
    glm::dvec3 v1 = glm::dvec3();
    glm::dvec3 v2 = glm::dvec3();
    glm::dvec3 normal = glm::dvec3();
    glm::dvec3 uInv = glm::dvec3();
    glm::dvec3 vInv = glm::dvec3();
    glm::dvec3 nInv = glm::dvec3();

    bool hasMaterials = false;
    Material materials[3];
};

class Trimesh {

public:
    explicit Trimesh(const Material &mat) : mat(mat) {
    }

    ~Trimesh();

    const Material &getMaterial() { return mat; }

    void setMaterial(const Material &m) { mat = m; }

    void addVertex(const glm::dvec3 &);

    void addMaterial(const Material &m);

    bool addFace(int a, int b, int c);

    std::vector<TrimeshFace *> getFaces() { return faces; };

    const char *doubleCheck();

    static Trimesh *fromBox(Material &mat, TransformNode *transform);

    static Trimesh *fromSquare(Material &mat, TransformNode *transform);

private:
    Trimesh *addVertices(glm::dvec3 a, glm::dvec3 b, glm::dvec3 c);

    Trimesh *addCube(Material &mat, glm::dvec3 *v);

    friend class TrimeshFace;

    Material mat;
    std::vector<glm::dvec3> vertices;
    std::vector<TrimeshFace *> faces;
    std::vector<Material> materials;
};