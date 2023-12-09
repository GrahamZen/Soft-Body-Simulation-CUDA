#include <softBody.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <glm/glm.hpp>
#include <bvh.cuh>
#include <simulationContext.h>


void SoftBody::PdSolver()
{
    Eigen::MatrixX3d fext;
    fext.resizeLike(model.positions());
    fext.setZero();
    // set gravity force
    fext.col(1).array() -= mcrpSimContext->GetGravity() * attrib.mass;
    if (!solver.ready())
    {
        solver.prepare(mcrpSimContext->GetDt());
    }
    solver.step(fext, mcrpSimContext->GetNumIterations());
    //fext.setZero();
}

void SoftBody::InitModel()
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi T, F;
    V.resize(numVerts, 3);
    for (int i = 0; i < numVerts; i++)
    {
        V.row(i) = Eigen::Vector3d(
            vertices[i].x,
            vertices[i].y,
            vertices[i].z);
        /**
        V(i, 0) = vertices[i].x;
        V(i, 1) = vertices[i].y;
        V(i, 2) = vertices[i].z;*/
    }

    // allocate space for triangles
    F.resize(numTets * 4, 3);
    // triangle indices
    for (int tet = 0; tet < numTets; tet++)
    {
        F(4 * tet, 0) = idx[tet * 4 + 0];
        F(4 * tet, 1) = idx[tet * 4 + 2];
        F(4 * tet, 2) = idx[tet * 4 + 1];
        F(4 * tet + 1, 0) = idx[tet * 4 + 0];
        F(4 * tet + 1, 1) = idx[tet * 4 + 3];
        F(4 * tet + 1, 2) = idx[tet * 4 + 2];
        F(4 * tet + 2, 0) = idx[tet * 4 + 0];
        F(4 * tet + 2, 1) = idx[tet * 4 + 1];
        F(4 * tet + 2, 2) = idx[tet * 4 + 3];
        F(4 * tet + 3, 0) = idx[tet * 4 + 1];
        F(4 * tet + 3, 1) = idx[tet * 4 + 2];
        F(4 * tet + 3, 2) = idx[tet * 4 + 3];
    }

    // allocate space for tetrahedra
    T.resize(numTets, 4);
    // tet indices
    int a, b, c, d;
    for (int i = 0; i < numTets; i++)
    {
        T(i, 0) = idx[i * 4 + 0];
        T(i, 1) = idx[i * 4 + 1];
        T(i, 2) = idx[i * 4 + 2];
        T(i, 3) = idx[i * 4 + 3];
    }

    Eigen::VectorXd masses(V.rows());
    masses.setConstant(attrib.mass);
    model = pd::deformable_mesh_t{ V, F, T, masses };
    model.constrain_deformation_gradient(attrib.stiffness_0);
    //model.velocity().rowwise() += Eigen::RowVector3d{ 0, 0, 0. };
    double const positional_wi = 1'000'000'000.;
    //model.constrain_deformation_gradient(deformation_gradient_wi);

    for (std::size_t i = 0u; i < attrib.numConstraints; ++i)
    {
        model.add_positional_constraint(i, positional_wi);
        model.fix(i);
    }
    solver.set_model(&model);
}

void SoftBody::setAttributes(GuiDataContainer::SoftBodyAttr& softBodyAttr)
{
    softBodyAttr.setJumpClean(jump);
    if (softBodyAttr.stiffness_0.second)
        attrib.stiffness_0 = softBodyAttr.stiffness_0.first;
    if (softBodyAttr.stiffness_1.second)
        attrib.stiffness_1 = softBodyAttr.stiffness_1.first;
}