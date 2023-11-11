#include "surfaceshader.h"


SurfaceShader::SurfaceShader()
    : ShaderProgram(),
    attrPos(-1), attrNor(-1), attrUV(-1),
    unifModel(-1), unifModelInvTr(-1), unifView(-1), unifProj(-1), unifCameraPos(-1)
{}

SurfaceShader::~SurfaceShader()
{}

void SurfaceShader::setupMemberVars()
{
    attrPos = glGetAttribLocation(prog, "vs_Pos");
    attrNor = glGetAttribLocation(prog, "vs_Nor");
    attrUV = glGetAttribLocation(prog, "vs_UV");

    unifModel = glGetUniformLocation(prog, "u_Model");
    unifModelInvTr = glGetUniformLocation(prog, "u_ModelInvTr");
    unifView = glGetUniformLocation(prog, "u_View");
    unifProj = glGetUniformLocation(prog, "u_Proj");
    unifCameraPos = glGetUniformLocation(prog, "u_CamPos");

    unifSampler2D = glGetUniformLocation(prog, "u_Texture");
    unifTime = glGetUniformLocation(prog, "u_Time");

    printGLErrorLog();
}

//This function, as its name implies, uses the passed in GL widget
void SurfaceShader::draw(Drawable& d, int textureSlot)
{
    useMe();

    if (unifSampler2D != -1)
    {
        glUniform1i(unifSampler2D, /*GL_TEXTURE*/textureSlot);
    }

    // Each of the following blocks checks that:
    //   * This shader has this attribute, and
    //   * This Drawable has a vertex buffer for this attribute.
    // If so, it binds the appropriate buffers to each attribute.

    if (attrPos != -1 && d.bindPos()) {
        glEnableVertexAttribArray(attrPos);
        glVertexAttribPointer(attrPos, 3, GL_FLOAT, false, 0, NULL);
    }

    if (attrNor != -1 && d.bindNor()) {
        glEnableVertexAttribArray(attrNor);
        glVertexAttribPointer(attrNor, 4, GL_FLOAT, false, 0, NULL);
    }

    if (attrUV != -1 && d.bindUV()) {
        glEnableVertexAttribArray(attrUV);
        glVertexAttribPointer(attrUV, 2, GL_FLOAT, false, 0, NULL);
    }

    // Bind the index buffer and then draw shapes from it.
    // This invokes the shader program, which accesses the vertex buffers.
    d.bindIdx();
    glDrawElements(d.drawMode(), d.elemCount(), GL_UNSIGNED_INT, 0);

    if (attrPos != -1) glDisableVertexAttribArray(attrPos);
    if (attrNor != -1) glDisableVertexAttribArray(attrNor);
    if (attrUV != -1) glDisableVertexAttribArray(attrUV);

    printGLErrorLog();
}


void SurfaceShader::setModelMatrix(const glm::mat4& model)
{
    useMe();

    if (unifModel != -1) {
        // Pass a 4x4 matrix into a uniform variable in our shader
                        // Handle to the matrix variable on the GPU
        glUniformMatrix4fv(unifModel,
            // How many matrices to pass
            1,
            // Transpose the matrix? OpenGL uses column-major, so no.
            GL_FALSE,
            // Pointer to the first element of the matrix
            &model[0][0]);

    }

    if (unifModelInvTr != -1) {
        glm::mat3 modelinvtr = glm::inverse(glm::transpose(glm::mat3(model)));
        // Pass a 4x4 matrix into a uniform variable in our shader
                        // Handle to the matrix variable on the GPU
        glUniformMatrix3fv(unifModelInvTr,
            // How many matrices to pass
            1,
            // Transpose the matrix? OpenGL uses column-major, so no.
            GL_FALSE,
            // Pointer to the first element of the matrix
            &modelinvtr[0][0]);
    }
}

void SurfaceShader::setViewProjMatrix(const glm::mat4& v, const glm::mat4& p)
{
    // Tell OpenGL to use this shader program for subsequent function calls
    useMe();

    if (unifView != -1) {
        // Pass a 4x4 matrix into a uniform variable in our shader
                        // Handle to the matrix variable on the GPU
        glUniformMatrix4fv(unifView,
            // How many matrices to pass
            1,
            // Transpose the matrix? OpenGL uses column-major, so no.
            GL_FALSE,
            // Pointer to the first element of the matrix
            &v[0][0]);
    }

    if (unifProj != -1) {
        // Pass a 4x4 matrix into a uniform variable in our shader
                        // Handle to the matrix variable on the GPU
        glUniformMatrix4fv(unifProj,
            // How many matrices to pass
            1,
            // Transpose the matrix? OpenGL uses column-major, so no.
            GL_FALSE,
            // Pointer to the first element of the matrix
            &p[0][0]);
    }
}

void SurfaceShader::setCameraPos(const glm::vec3& v) {
    useMe();

    if (unifCameraPos != -1) {
        // Pass a 4x4 matrix into a uniform variable in our shader
                        // Handle to the matrix variable on the GPU
        glUniform3fv(unifCameraPos,
            // How many vectors to pass
            1,
            // Pointer to the first element of the matrix
            &v[0]);

    }

}

