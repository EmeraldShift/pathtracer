#pragma once

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <iostream>


struct f4 {
    float4 vec;

    __host__ __device__
    f4() : vec({0, 0, 0, 0}) {}

    __host__ __device__
    f4(float xyzw) : vec({xyzw, xyzw, xyzw, xyzw}) {}

    __host__ __device__
    f4(float x, float y, float z) : vec({x, y, z, 0}) {}

    __host__ __device__
    f4(float x, float y, float z, float w) : vec({x, y, z, w}) {}

    __host__ __device__
    f4(const glm::vec3 &v) : vec({v.x, v.y, v.z, 0}) {}

    __host__ __device__
    f4(const glm::vec4 &v) : vec({v.x, v.y, v.z, v.w}) {}

    __host__ __device__
    float &operator[](unsigned long long n) {
        return n == 0 ? vec.x : n == 1 ? vec.y : n == 2 ? vec.z : vec.w;
    }

    __host__ __device__
    const float &operator[](unsigned long long n) const {
        return n == 0 ? vec.x : n == 1 ? vec.y : n == 2 ? vec.z : vec.w;
    }
};

// Negation
inline __host__ __device__ f4 operator-(const f4 &a) {
    return {-a[0], -a[1], -a[2], -a[3]};
}

// Addition
inline __host__ __device__ f4 operator+(const f4 &a, const f4 &b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}

inline __host__ __device__ f4 operator+(const f4 &a, const float &f) {
    return {a[0] + f, a[1] + f, a[2] + f, a[3] + f};
}

inline __host__ __device__ f4 operator+(const float &f, const f4 &a) {
    return {f + a[0], f + a[1], f + a[2], f + a[3]};
}

inline __host__ __device__ f4 operator+=(f4 &a, const float &f) {
    a[0] += f;
    a[1] += f;
    a[2] += f;
    a[3] += f;
    return a;
}

inline __host__ __device__ f4 operator+=(f4 &a, const f4 &b) {
    a[0] += b[0];
    a[1] += b[1];
    a[2] += b[2];
    a[3] += b[3];
    return a;
}

// Subtraction
inline __host__ __device__ f4 operator-(const f4 &a, const f4 &b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
}

inline __host__ __device__ f4 operator-(const f4 &a, const float &f) {
    return {a[0] - f, a[1] - f, a[2] - f, a[3] - f};
}

inline __host__ __device__ f4 operator-(const float &f, const f4 &a) {
    return {f - a[0], f - a[1], f - a[2], f - a[3]};
}

inline __host__ __device__ f4 operator-=(f4 &a, const float &f) {
    a[0] -= f;
    a[1] -= f;
    a[2] -= f;
    a[3] -= f;
    return a;
}

inline __host__ __device__ f4 operator-=(f4 &a, const f4 &b) {
    a[0] -= b[0];
    a[1] -= b[1];
    a[2] -= b[2];
    a[3] -= b[3];
    return a;
}

// Multiplication
inline __host__ __device__ f4 operator*(const f4 &a, const f4 &b) {
    return {a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]};
}

inline __host__ __device__ f4 operator*(const f4 &a, const float &f) {
    return {a[0] * f, a[1] * f, a[2] * f, a[3] * f};
}

inline __host__ __device__ f4 operator*(const float &f, const f4 &a) {
    return {f * a[0], f * a[1], f * a[2], f * a[3]};
}

inline __host__ __device__ f4 operator*=(f4 &a, const float &f) {
    a[0] *= f;
    a[1] *= f;
    a[2] *= f;
    a[3] *= f;
    return a;
}

inline __host__ __device__ f4 operator*=(f4 &a, const f4 &b) {
    a[0] *= b[0];
    a[1] *= b[1];
    a[2] *= b[2];
    a[3] *= b[3];
    return a;
}

// Division
inline __host__ __device__ f4 operator/(const f4 &a, const f4 &b) {
    return {a[0] / b[0], a[1] / b[1], a[2] / b[2], a[3] / b[3]};
}

inline __host__ __device__ f4 operator/(const f4 &a, const float &f) {
    return {a[0] / f, a[1] / f, a[2] / f, a[3] / f};
}

inline __host__ __device__ f4 operator/(const float &f, const f4 &a) {
    return {f / a[0], f / a[1], f / a[2], f / a[3]};
}

inline __host__ __device__ f4 operator/=(f4 &a, const float &f) {
    a[0] /= f;
    a[1] /= f;
    a[2] /= f;
    a[3] /= f;
    return a;
}

inline __host__ __device__ f4 operator/=(f4 &a, const f4 &b) {
    a[0] /= b[0];
    a[1] /= b[1];
    a[2] /= b[2];
    a[3] /= b[3];
    return a;
}

inline std::ostream &operator<<(std::ostream &o, const f4 &a) {
    return o << "[ " << a[0] << ", " << a[1] << ", " << a[2] << ", " << a[3] << " ]";
}

namespace f4m {
    inline __host__ __device__ f4 min(const f4 &a, const f4 &b) {
        return {
                fminf(a[0], b[0]),
                fminf(a[1], b[1]),
                fminf(a[2], b[2]),
                fminf(a[3], b[3])
        };
    }

    inline __host__ __device__ f4 max(const f4 &a, const f4 &b) {
        return {
                fmaxf(a[0], b[0]),
                fmaxf(a[1], b[1]),
                fmaxf(a[2], b[2]),
                fmaxf(a[3], b[3])
        };
    }

    inline __host__ __device__ f4 clamp(const f4 &a, const float &lo, const float &hi) {
        return min(max(a, lo), hi);
    }

    inline __host__ __device__ float dot(const f4 &a, const f4 &b) {
        return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    }

    inline __host__ __device__ float length(const f4 &a) {
        return sqrtf(dot(a, a));
    }

    inline __host__ __device__ float length2(const f4 &a) {
        return dot(a, a);
    }

    inline __host__ __device__ f4 normalize(const f4 &a) {
        return a * rsqrtf(dot(a, a));
    }

    inline __host__ __device__ f4 cross(const f4 &a, const f4 &b) {
        return {a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]};
    }
}
