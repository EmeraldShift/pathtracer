#include "light.h"

double PointLight::distanceAttenuation(const glm::dvec3& p) const
{
    auto dist = getDistanceFrom(p);
    auto dat = constantTerm + linearTerm * dist + quadraticTerm * dist * dist;
    return dat > 1 ? 1.0/dat : 1;
}

glm::dvec3 PointLight::getColor() const
{
	return color;
}

glm::dvec3 PointLight::getDirection(const glm::dvec3& P) const
{
	return glm::normalize(position - P);
}

glm::dvec3 PointLight::shadowAttenuation(const ray& r, const glm::dvec3& p) const
{
    // TODO demolish - ain't no shadow rays in path tracing!
	return glm::dvec3(1,1,1);
}

double PointLight::getDistanceFrom(const glm::dvec3 &p) const {
    return glm::length(position - p);
}

#define VERBOSE 0

