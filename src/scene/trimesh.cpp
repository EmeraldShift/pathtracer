#include "trimesh.h"

#include <cmath>

TrimeshFace *TrimeshFace::create(const Material &mat, Trimesh *parent, int a, int b, int c) {
    // Compute the face normal here, not on the fly
    glm::dvec3 a_coords = parent->vertices[a];
    glm::dvec3 b_coords = parent->vertices[b];
    glm::dvec3 c_coords = parent->vertices[c];

    glm::dvec3 vab = (b_coords - a_coords);
    glm::dvec3 vac = (c_coords - a_coords);
    glm::dvec3 vcb = (b_coords - c_coords);

    // If this is a degenerate face, quit
    if (glm::length(vab) == 0.0 || glm::length(vac) == 0.0 ||
        glm::length(vcb) == 0.0)
        return nullptr;

    auto face = new TrimeshFace();
    auto normal = glm::normalize(glm::cross(vab, vac));
    auto inverse = glm::inverse(glm::dmat3(vab, vac, normal));
    face->mat = mat;
    face->parent = parent;
    face->ids[0] = a;
    face->ids[1] = b;
    face->ids[2] = c;
    face->normal = normal;
    face->uInv = glm::dvec3(inverse[0][0], inverse[1][0], inverse[2][0]);
    face->vInv = glm::dvec3(inverse[0][1], inverse[1][1], inverse[2][1]);
    face->nInv = glm::dvec3(inverse[0][2], inverse[1][2], inverse[2][2]);
    face->bounds = face->computeLocalBoundingBox();
    return face;
}

BoundingBox TrimeshFace::computeLocalBoundingBox() {
    BoundingBox bb;
    bb.setMax(glm::max(parent->vertices[ids[0]],
                       parent->vertices[ids[1]]));
    bb.setMin(glm::min(parent->vertices[ids[0]],
                       parent->vertices[ids[1]]));

    bb.setMax(glm::max(parent->vertices[ids[2]],
                       bb.getMax()));
    bb.setMin(glm::min(parent->vertices[ids[2]],
                       bb.getMin()));
    return bb;
}

Trimesh::~Trimesh() {
    for (auto f : faces)
        delete f;
}

void Trimesh::addVertex(const glm::dvec3 &v) {
    vertices.emplace_back(v);
}

void Trimesh::addMaterial(const Material &m) {
    materials.emplace_back(m);
}

// Returns false if the vertices a,b,c don't all exist
bool Trimesh::addFace(int a, int b, int c) {
    int vCount = vertices.size();

    if (a >= vCount || b >= vCount || c >= vCount)
        return false;

    auto face = TrimeshFace::create(mat, this, a, b, c);
    if (face)
        faces.push_back(face);

    // Don't add faces to the scene's object list so we can cull by bounding
    // box
    return true;
}

// Check to make sure that if we have per-vertex materials or normals
// they are the right number.
const char *Trimesh::doubleCheck() {
    if (!materials.empty() && materials.size() != vertices.size())
        return "Bad Trimesh: Wrong number of materials.";
    return nullptr;
}

bool TrimeshFace::intersect(ray &r, isect &i) const {
    auto p = r.getPosition() - parent->vertices[ids[0]];
    auto d = r.getDirection();
    auto p_z = glm::dot(nInv, p);
    auto d_z = glm::dot(nInv, d);
    if (p_z < 0 == d_z < 0)
        return false;

    auto t = -p_z / d_z;
    if (t > i.getT())
        return false;

    auto hit = p + t * d;
    double u = glm::dot(uInv, hit); // p1
    double v = glm::dot(vInv, hit); // p2;
    double w = 1 - u - v; // p0

    if (u < 0 || u > 1 || v < 0 || v > 1 || u + v > 1)
        return false;

    i.setT(t);
    i.setN(normal);
    i.setUVCoordinates(glm::dvec2(u, v));
    if (parent->materials.empty()) {
        i.setMaterial(mat);
    } else {
        Material m0 = parent->materials[ids[0]];
        Material m1 = parent->materials[ids[1]];
        Material m2 = parent->materials[ids[2]];
        m0 = w * m0;
        m1 = u * m1;
        m2 = v * m2;
        m0 += m1;
        m0 += m2;
        i.setMaterial(m0);
    }
    return true;
}


Trimesh *Trimesh::addVertices(glm::dvec3 a, glm::dvec3 b, glm::dvec3 c) {
    addVertex(a);
    addVertex(b);
    addVertex(c);
    return this;
}

Trimesh *Trimesh::addCube(Material &mat, glm::dvec3 *v) {
    addVertices(v[0], v[1], v[2]);
    addVertices(v[3], v[2], v[1]);
    addVertices(v[4], v[6], v[5]);
    addVertices(v[7], v[5], v[6]);
    addVertices(v[0], v[2], v[4]);
    addVertices(v[6], v[4], v[2]);
    addVertices(v[1], v[5], v[3]);
    addVertices(v[7], v[3], v[5]);
    addVertices(v[0], v[1], v[4]);
    addVertices(v[5], v[4], v[1]);
    addVertices(v[2], v[6], v[3]);
    addVertices(v[7], v[3], v[6]);
    for (int i = 0; i < 12; i++)
        addFace(i * 3, i * 3 + 1, i * 3 + 2);
    return this;
}

Trimesh *Trimesh::fromBox(Material &mat, TransformNode *transform) {
    auto p000 = transform->localToGlobalCoords(glm::dvec3(-0.5, -0.5, -0.5));
    auto p001 = transform->localToGlobalCoords(glm::dvec3(-0.5, -0.5, 0.5));
    auto p010 = transform->localToGlobalCoords(glm::dvec3(-0.5, 0.5, -0.5));
    auto p011 = transform->localToGlobalCoords(glm::dvec3(-0.5, 0.5, 0.5));
    auto p100 = transform->localToGlobalCoords(glm::dvec3(0.5, -0.5, -0.5));
    auto p101 = transform->localToGlobalCoords(glm::dvec3(0.5, -0.5, 0.5));
    auto p110 = transform->localToGlobalCoords(glm::dvec3(0.5, 0.5, -0.5));
    auto p111 = transform->localToGlobalCoords(glm::dvec3(0.5, 0.5, 0.5));
    glm::dvec3 vertices[8] = {p000, p001, p010, p011, p100, p101, p110, p111};
    return (new Trimesh(mat))->addCube(mat, vertices);
}

Trimesh *Trimesh::fromSquare(Material &mat, TransformNode *transform) {
    auto p000 = transform->localToGlobalCoords(glm::dvec3(-0.5, -0.5, -0.001));
    auto p001 = transform->localToGlobalCoords(glm::dvec3(-0.5, -0.5, 0.001));
    auto p010 = transform->localToGlobalCoords(glm::dvec3(-0.5, 0.5, -0.001));
    auto p011 = transform->localToGlobalCoords(glm::dvec3(-0.5, 0.5, 0.001));
    auto p100 = transform->localToGlobalCoords(glm::dvec3(0.5, -0.5, -0.001));
    auto p101 = transform->localToGlobalCoords(glm::dvec3(0.5, -0.5, 0.001));
    auto p110 = transform->localToGlobalCoords(glm::dvec3(0.5, 0.5, -0.001));
    auto p111 = transform->localToGlobalCoords(glm::dvec3(0.5, 0.5, 0.001));
    glm::dvec3 vertices[8] = {p000, p001, p010, p011, p100, p101, p110, p111};
    return (new Trimesh(mat))->addCube(mat, vertices);
}
