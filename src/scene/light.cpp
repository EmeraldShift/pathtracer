#include <cmath>
#include <iostream>

#include "light.h"
#include <glm/glm.hpp>
#include <glm/gtx/io.hpp>


using namespace std;

double DirectionalLight::distanceAttenuation(const glm::dvec3& P) const
{
	// distance to light is infinite, so f(di) goes to 0.  Return 1.
	return 1.0;
}


glm::dvec3 DirectionalLight::shadowAttenuation(const ray& r, const glm::dvec3& p) const
{
	// YOUR CODE HERE:
	// You should implement shadow-handling code here.
	return glm::dvec3(1.0, 1.0, 1.0);
}

glm::dvec3 DirectionalLight::getColor() const
{
	return color;
}

glm::dvec3 DirectionalLight::getDirection(const glm::dvec3& P) const
{
	return -orientation;
}

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

double DirectionalLight::getDistanceFrom(const glm::dvec3 &p) const {
    return 10;
}

#define VERBOSE 0

