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
in vec4 vs_Col;

in vec2 vs_UV;// The array of vertex texture coordinates passed to the shader

out vec4 fs_Nor;// The array of normals that has been transformed by u_ModelInvTr. This is implicitly passed to the fragment shader.
out vec4 fs_Col;
out vec4 fs_LightVec;// The direction in which our virtual light lies, relative to each vertex. This is implicitly passed to the fragment shader.
out vec2 fs_UV;// The UV of each vertex. This is implicitly passed to the fragment shader.

void main()
{
    fs_Col=vs_Col;
    fs_UV=vs_UV;// Pass the vertex UVs to the fragment shader for interpolation
    
    fs_Nor=normalize(vec4(u_ModelInvTr*vec3(vs_Nor),0));// Pass the vertex normals to the fragment shader for interpolation.
    // Transform the geometry's normals by the inverse transpose of the
    // model matrix. This is necessary to ensure the normals remain
    // perpendicular to the surface after the surface is transformed by
    // the model matrix.
    
    vec4 modelposition=u_Model*vec4(vs_Pos,1.0);// Temporarily store the transformed vertex positions for use below
    
    fs_LightVec=(inverse(u_View)*vec4(0,0,0,1))-modelposition;// Compute the direction in which the light source lies
    
    gl_Position=u_Proj*u_View*modelposition;// gl_Position is a built-in variable of OpenGL which is
    // used to render the final positions of the geometry's vertices
}
