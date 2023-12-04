#version 150
// ^ Change this to version 130 if you have compatibility issues

//This is a vertex shader. While it is called a "shader" due to outdated conventions, this file
//is used to apply matrix transformations to the arrays of vertex data passed to it.
//Since this code is run on your GPU, each vertex is transformed simultaneously.
//If it were run on your CPU, each vertex would have to be processed in a FOR loop, one at a time.
//This simultaneous transformation allows your program to run much faster, especially when rendering
//geometry with millions of vertices.

uniform mat4 u_Model;// The matrix that defines the transformation of the
// object we're rendering. In this assignment,
// this will be the result of traversing your scene graph.

uniform mat3 u_ModelInvTr;// The inverse transpose of the model matrix.
// This allows us to transform the object's normals properly
// if the object has been non-uniformly scaled.

uniform mat4 u_View;// The matrix that defines the camera's transformation.
uniform mat4 u_Proj;// The matrix that defines the camera's projection.

in vec3 vs_Pos;// The array of vertex positions passed to the shader

in vec4 vs_Nor;// The array of vertex normals passed to the shader

in vec2 vs_UV;// The array of vertex texture coordinates passed to the shader
in vec4 vs_Col;// The array of vertex colors passed to the shader

out vec4 fs_Col;// The color of each vertex. This is implicitly passed to the fragment shader.

void main()
{        
    vec4 modelposition=u_Model*vec4(vs_Pos,1.0);// Temporarily store the transformed vertex positions for use below
    
    fs_Col = vs_Col;// Pass the vertex colors to the fragment shader for interpolation
    gl_Position=u_Proj*u_View*modelposition;// gl_Position is a built-in variable of OpenGL which is
    // used to render the final positions of the geometry's vertices
}
