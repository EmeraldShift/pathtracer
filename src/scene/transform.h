#pragma once

inline glm::vec3 operator*(const glm::mat4x4 &mat, const glm::vec3 &vec) {
    glm::vec4 vec4(vec[0], vec[1], vec[2], 1.0);
    auto ret = mat * vec4;
    return glm::vec3(ret[0], ret[1], ret[2]);
}

class TransformNode {
public:
    ~TransformNode() {
        for (auto c : children)
            delete c;
    }

    TransformNode *createChild(const glm::mat4x4 &txf) {
        auto child = new TransformNode(this, txf);
        children.push_back(child);
        return child;
    }

    glm::vec3 localToGlobalCoords(const glm::vec3 &v) {
        return xform * v;
    }

    glm::vec4 localToGlobalCoords(const glm::vec4 &v) {
        return xform * v;
    }

    glm::vec3 localToGlobalCoordsNormal(const glm::vec3 &v) {
        return glm::normalize(normi * v);
    }

    const glm::dmat4x4 &transform() const { return xform; }

protected:
    TransformNode(TransformNode *parent, const glm::mat4x4 &xform) {
        this->parent = parent;
        if (parent == nullptr)
            this->xform = xform;
        else
            this->xform = parent->xform * xform;
        inverse = glm::inverse(this->xform);
        normi = glm::transpose(glm::inverse(glm::mat3x3(this->xform)));
    }

    glm::mat4x4 xform;
    glm::mat4x4 inverse;
    glm::mat3x3 normi;

    TransformNode *parent;
    std::vector<TransformNode *> children;
};

class TransformRoot : public TransformNode {
public:
    TransformRoot() : TransformNode(nullptr, glm::mat4x4(1.0)) {}
};