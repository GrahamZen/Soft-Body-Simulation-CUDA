#version 330 core

in vec3 fs_Pos;

out vec4 out_Col;

uniform samplerCube u_EnvironmentMap;

void main() {
    vec3 envColor = texture(u_EnvironmentMap, fs_Pos).rgb;
    // Uncomment the line below if you want to see different mip levels
    // of your glossy irradiance map. Change 1.2 to anything up to 4
    // to see the different mip levels.
//    vec3 envColor = textureLod(u_EnvironmentMap, fs_Pos, 1.2).rgb;

    // Reinhard op + gamma correction
    // envColor = envColor / (envColor + vec3(1.0));
    envColor = pow(envColor, vec3(1.0/2.2));

    out_Col = vec4(envColor, 1.0);
}
