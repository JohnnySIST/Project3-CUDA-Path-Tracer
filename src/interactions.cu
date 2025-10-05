#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng,
    bool outside)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    if (m.hasReflective > 0.5f) { // reflective material
        // ensure material is double-sided
        float cosTheta = glm::dot(normal, -pathSegment.ray.direction);
        if (cosTheta < 0.0f) {
            cosTheta = -cosTheta;
            normal = -normal;
        }

        glm::vec3 reflectDir = glm::reflect(pathSegment.ray.direction, normal);
        pathSegment.ray.origin = intersect;
        pathSegment.ray.direction = reflectDir;
        pathSegment.color *= m.color;
        pathSegment.remainingBounces--;
        return;
    } else if (m.hasRefractive > 0.5f) { // refractive material
        glm::vec3 inDir = pathSegment.ray.direction;
        float eta = m.indexOfRefraction;
        float cosTheta = glm::dot(normal, -inDir);
        if (cosTheta < 0.0f) {
            eta = 1.0f / eta;
            cosTheta = -cosTheta;
            normal = -normal;
        }
        /* following https://pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission
        float sinThetaSquare = fmaxf(0.0f, 1.0f - cosTheta * cosTheta);
        float sinTheta2Square = sinThetaSquare / (eta * eta);
        if (sinTheta2Square >= 1.0f) {
            pathSegment.ray.direction = glm::reflect(inDir, n);
        } else {
            float cosTheta2 = sqrtf(1.0f - sinTheta2Square);
            pathSegment.ray.direction = (-inDir) / eta + (cosTheta / eta - cosTheta2) * n;
        } */
        float R0 = (eta - 1) * (eta - 1) / ((eta + 1) * (eta + 1));
        float schlickProb = R0 + (1 - R0) * powf(1 - cosTheta, 5);
        thrust::uniform_real_distribution<float> u01(0, 1);
        // Schlick's approximation
        if (u01(rng) > schlickProb) {
            glm::vec3 refractDir = glm::refract(inDir, normal, eta);
            if (glm::length(refractDir) > 0.0f) {
                pathSegment.ray.direction = refractDir;
            } else {
                pathSegment.ray.direction = glm::reflect(inDir, normal);
            }
        } else {
            pathSegment.ray.direction = glm::reflect(inDir, normal);
        }
        pathSegment.ray.origin = intersect + 0.001f * pathSegment.ray.direction;
        pathSegment.color *= m.color;
        pathSegment.remainingBounces--;
        return;
    } else { // diffuse material
        // ensure material is double-sided
        float cosTheta = glm::dot(normal, -pathSegment.ray.direction);
        if (cosTheta < 0.0f) {
            cosTheta = -cosTheta;
            normal = -normal;
        }

        glm::vec3 scatterDir = calculateRandomDirectionInHemisphere(normal, rng);
        pathSegment.ray.origin = intersect;
        pathSegment.ray.direction = scatterDir;
        pathSegment.color *= m.color;
        pathSegment.remainingBounces--;
    }
    return;
}
