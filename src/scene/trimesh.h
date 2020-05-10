#pragma once

#include "geometry.h"
#include "material.h"
#include "transform.h"

class Trimesh;

class TrimeshFace : public Geometry {
public:
    glm::vec3 getNormal() { return normal; }

    __host__ __device__ static bool intersect(void *obj, ray &r, isect &i);

    TrimeshFace *clone() const override;

    static TrimeshFace *create(const Material &mat, Trimesh *parent, int a, int b, int c);

private:
    TrimeshFace() = default;

    BoundingBox computeLocalBoundingBox();

    glm::vec3 vertices[3];
    glm::vec3 v0 = glm::vec3();
    glm::vec3 v1 = glm::vec3();
    glm::vec3 v2 = glm::vec3();
    glm::vec3 normal = glm::vec3();
    glm::vec3 uInv = glm::vec3();
    glm::vec3 vInv = glm::vec3();
    glm::vec3 nInv = glm::vec3();

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

    void addVertex(const glm::vec3 &);

    void addMaterial(const Material &m);

    bool addFace(int a, int b, int c);

    std::vector<TrimeshFace *> getFaces() { return faces; };

    const char *doubleCheck();

    static Trimesh *fromBox(Material &mat, TransformNode *transform);

    static Trimesh *fromSquare(Material &mat, TransformNode *transform);

private:
    Trimesh *addVertices(glm::vec3 a, glm::vec3 b, glm::vec3 c);

    Trimesh *addCube(glm::vec3 *v);

    friend class TrimeshFace;

    Material mat;
    std::vector<glm::vec3> vertices;
    std::vector<TrimeshFace *> faces;
    std::vector<Material> materials;
};