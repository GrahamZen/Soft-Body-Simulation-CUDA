#version 330

//This is a fragment shader. If you've opened this file first, please open and read lambert.vert.glsl before reading on.
//Unlike the vertex shader, the fragment shader actually does compute the shading of geometry.
//For every pixel in your program's output screen, the fragment shader is run for every bit of geometry that particular pixel overlaps.
//By implicitly interpolating the position data passed into the fragment shader by the vertex shader, the fragment shader
//can compute what color to apply to its pixel based on things like vertex position, light position, and vertex color.

in vec4 fs_Col;
in float vertexIndex;

layout(location=0)out vec4 out_Col;//This is the final output color that you will see on your screen for the pixel that is currently being processed.

void main()
{
    if (fs_Col.a < 0.1)
    {
        discard;
    }
    out_Col = vec4(fs_Col.r, fs_Col.g * fs_Col.a, fs_Col.b * fs_Col.a, fs_Col.a);
}
