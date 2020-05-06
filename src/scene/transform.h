#pragma once

inline glm::dvec3 operator*(const glm::dmat4x4 &mat, const glm::dvec3 &vec) {
    glm::dvec4 vec4(vec[0], vec[1], vec[2], 1.0);
    auto ret = mat * vec4;
    return glm::dvec3(ret[0], ret[1], ret[2]);
}

class TransformNode {
public:
    ~TransformNode() {
        for (auto c : children)
            delete c;
    }

    TransformNode *createChild(const glm::dmat4x4 &txf) {
        auto child = new TransformNode(this, txf);
        children.push_back(child);
        return child;
    }

    glm::dvec3 localToGlobalCoords(const glm::dvec3 &v) {
        return xform * v;
    }

    glm::dvec4 localToGlobalCoords(const glm::dvec4 &v) {
        return xform * v;
    }

    glm::dvec3 localToGlobalCoordsNormal(const glm::dvec3 &v) {
        return glm::normalize(normi * v);
    }

    const glm::dmat4x4 &transform() const { return xform; }

protected:
    TransformNode(TransformNode *parent, const glm::dmat4x4 &xform) {
        this->parent = parent;
        if (parent == nullptr)
            this->xform = xform;
        else
            this->xform = parent->xform * xform;
        inverse = glm::inverse(this->xform);
        normi = glm::transpose(glm::inverse(glm::dmat3x3(this->xform)));
    }

    glm::dmat4x4 xform;
    glm::dmat4x4 inverse;
    glm::dmat3x3 normi;

    TransformNode *parent;
    std::vector<TransformNode *> children;
};

class TransformRoot : public TransformNode {
public:
    TransformRoot() : TransformNode(nullptr, glm::dmat4x4(1.0)) {}
};