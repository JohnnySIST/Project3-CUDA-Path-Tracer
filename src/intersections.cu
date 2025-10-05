#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        // normal = -normal; // don't need to flip normal for refractive object
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float meshIntersectionTest(
    MeshGPU mesh,
    Geom model,
    Ray r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    float t = -1;
    for (int i = 0; i < mesh.numTriangles; ++i) {
        glm::vec3 v0 = mesh.positions[i * 3 + 0];
        glm::vec3 v1 = mesh.positions[i * 3 + 1];
        glm::vec3 v2 = mesh.positions[i * 3 + 2];
        
        Ray q;
        q.origin = multiplyMV(model.inverseTransform, glm::vec4(r.origin, 1.0f));
        q.direction = glm::normalize(multiplyMV(model.inverseTransform, glm::vec4(r.direction, 0.0f)));
        
        glm::vec3 bary;
        bool hit = glm::intersectRayTriangle(q.origin, q.direction, v0, v1, v2, bary);
        if (hit && bary.z > 0.01f) {
            t = bary.z;
            glm::vec3 objspaceIntersection = getPointOnRay(q, t);
            intersectionPoint = multiplyMV(model.transform, glm::vec4(objspaceIntersection, 1.0f));

            if (true || mesh.normals == nullptr) {
                glm::vec3 e0 = v1 - v0;
                glm::vec3 e1 = v2 - v0;
                normal = glm::normalize(multiplyMV(model.invTranspose, glm::vec4(glm::cross(e0, e1), 0.f)));
                outside = glm::dot(normal, r.direction) < 0;
                if (!outside) {
                    normal = -normal;
                }
            } else {
                float w = 1.0f - bary.x - bary.y;
                glm::vec3 n0 = mesh.normals[i * 3 + 0];
                glm::vec3 n1 = mesh.normals[i * 3 + 1];
                glm::vec3 n2 = mesh.normals[i * 3 + 2];
                glm::vec3 n = w * n0 + bary.x * n1 + bary.y * n2;
                normal = glm::normalize(multiplyMV(model.invTranspose, glm::vec4(n, 0.f)));
                outside = glm::dot(normal, r.direction) < 0;
            }
        }
    }
    return t;
}