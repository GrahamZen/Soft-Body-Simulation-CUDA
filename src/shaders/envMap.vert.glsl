#version 330 core

// A shader program used to render the environment map as
// a background image in the 3D scene. Notably, the last
// line of code sets our vertices' screen-space Z coords
// to 1.0 so that they are always the farthest element
// from the camera.
// We use glDepthFunc(GL_LEQUAL) CPU-side to allow Z = 1
// to overwrite the depth buffer.

uniform mat4 u_View;// The matrix that defines the camera's transformation.
uniform mat4 u_Proj;// The matrix that defines the camera's projection.

in vec3 vs_Pos;
out vec3 fs_Pos;

void main() {
    fs_Pos = vs_Pos;
    vec4 clipPos = u_Proj * u_View * vec4(vs_Pos, 1.0);

    gl_Position = clipPos.xyww;
}
