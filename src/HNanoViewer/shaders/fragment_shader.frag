#version 330 core

in vec3 vPosition;
out vec4 FragColor;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
uniform sampler3D volumeTexture;

uniform vec3 cameraPos;

const int MAX_STEPS = 256;
const float STEP_SIZE = 0.01;

void main()
{
    // Transformations
    mat4 invModel = inverse(model);
    vec3 rayOrigin = vec3(invModel * vec4(cameraPos, 1.0));
    vec3 rayDir = normalize(vPosition - rayOrigin);

    // Compute intersection with the volume bounding box
    vec3 boxMin = vec3(-0.5);
    vec3 boxMax = vec3(0.5);
    float tMin, tMax;

    vec3 invDir = 1.0 / rayDir;
    vec3 t0s = (boxMin - rayOrigin) * invDir;
    vec3 t1s = (boxMax - rayOrigin) * invDir;

    vec3 tSmaller = min(t0s, t1s);
    vec3 tLarger = max(t0s, t1s);

    tMin = max(max(tSmaller.x, tSmaller.y), max(tSmaller.z, 0.0));
    tMax = min(min(tLarger.x, tLarger.y), tLarger.z);

    if (tMin >= tMax)
    discard; // No intersection

    // Ray marching
    float t = tMin;
    vec3 pos;
    vec4 accumulatedColor = vec4(0.0);
    for (int i = 0; i < MAX_STEPS && t < tMax; ++i)
    {
        pos = rayOrigin + t * rayDir;
        float density = texture(volumeTexture, pos + vec3(0.5)).r;

        // Simple transfer function (map density to color)
        vec4 colorSample = vec4(density);

        // Accumulate color using alpha blending
        accumulatedColor.rgb += (1.0 - accumulatedColor.a) * colorSample.rgb * colorSample.a;
        accumulatedColor.a += (1.0 - accumulatedColor.a) * colorSample.a;

        if (accumulatedColor.a >= 0.95)
        break;

        t += STEP_SIZE;
    }

    FragColor = accumulatedColor;

    // Discard fragments with negligible opacity
    if (FragColor.a <= 0.001)
    discard;
}
