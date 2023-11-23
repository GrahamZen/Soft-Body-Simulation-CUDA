#include "deformation_gradient_constraint.h"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <array>
#include <iostream>

#include <glm/glm.hpp>
#include <svd3.h>

namespace pd {

deformation_gradient_constraint_t::deformation_gradient_constraint_t(
    std::initializer_list<index_type> indices,
    scalar_type wi,
    positions_type const& p)
    : base_type(indices, wi), V0_{0.}, DmInv_{}
{
    assert(indices.size() == 4u);

    auto const v1 = this->indices().at(0);
    auto const v2 = this->indices().at(1);
    auto const v3 = this->indices().at(2);
    auto const v4 = this->indices().at(3);

    auto const p1 = p.row(v1);
    auto const p2 = p.row(v2);
    auto const p3 = p.row(v3);
    auto const p4 = p.row(v4);

    Eigen::Matrix3d Dm;
    Dm.col(0) = (p1 - p4).transpose();
    Dm.col(1) = (p2 - p4).transpose();
    Dm.col(2) = (p3 - p4).transpose();

    V0_    = (1. / 6.) * Dm.determinant();
    DmInv_ = Dm.inverse();
}

deformation_gradient_constraint_t::scalar_type
deformation_gradient_constraint_t::evaluate(positions_type const& p, masses_type const& M)
{
    auto const v1 = this->indices().at(0);
    auto const v2 = this->indices().at(1);
    auto const v3 = this->indices().at(2);
    auto const v4 = this->indices().at(3);

    Eigen::Vector3d const p1 = p.row(v1).transpose();
    Eigen::Vector3d const p2 = p.row(v2).transpose();
    Eigen::Vector3d const p3 = p.row(v3).transpose();
    Eigen::Vector3d const p4 = p.row(v4).transpose();

    Eigen::Matrix3d Ds;
    Ds.col(0) = p1 - p4;
    Ds.col(1) = p2 - p4;
    Ds.col(2) = p3 - p4;

    scalar_type const Vsigned = (1. / 6.) * Ds.determinant();

    bool const is_V_positive  = Vsigned >= 0.;
    bool const is_V0_positive = V0_ >= 0.;
    bool const is_tet_inverted =
        (is_V_positive && !is_V0_positive) || (!is_V_positive && is_V0_positive);

    Eigen::Matrix3d const F = Ds * DmInv_;
    Eigen::Matrix3d const I = Eigen::Matrix3d::Identity();

    Eigen::JacobiSVD<Eigen::Matrix3d> UFhatV(F, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d const Fsigma = UFhatV.singularValues();
    Eigen::Matrix3d Fhat;
    Fhat.setZero();
    Fhat(0, 0) = Fsigma(0);
    Fhat(1, 1) = Fsigma(1);
    Fhat(2, 2) = Fsigma(2);

    Eigen::Matrix3d U       = UFhatV.matrixU();
    Eigen::Matrix3d const V = UFhatV.matrixV();

    if (is_tet_inverted)
    {
        Fhat(2, 2) = -Fhat(2, 2);
        U.col(2)   = -U.col(2);
    }

    // stress reaches maximum at 58% compression
    scalar_type constexpr min_singular_value = 0.577;
    Fhat(0, 0)                               = std::max(Fhat(0, 0), min_singular_value);
    Fhat(1, 1)                               = std::max(Fhat(1, 1), min_singular_value);
    Fhat(2, 2)                               = std::max(Fhat(2, 2), min_singular_value);

    scalar_type const young_modulus = 1'000'000'000.;
    scalar_type const poisson_ratio = 0.45;
    scalar_type const mu            = (young_modulus) / (2. * (1 + poisson_ratio));
    scalar_type const lambda =
        (young_modulus * poisson_ratio) / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio));

    Eigen::Matrix3d const Ehat     = 0.5 * (Fhat.transpose() * Fhat - I);
    scalar_type const EhatTrace    = Ehat.trace();
    Eigen::Matrix3d const Piolahat = Fhat * ((2. * mu * Ehat) + (lambda * EhatTrace * I));

    Eigen::Matrix3d const E  = U * Ehat * V.transpose();
    scalar_type const Etrace = E.trace();
    scalar_type const psi    = mu * (E.array() * E.array()).sum() + 0.5 * lambda * Etrace * Etrace;

    scalar_type const V0 = std::abs(V0_);
    scalar_type const C = V0 * psi;

    return C;
}

void deformation_gradient_constraint_t::project_wi_SiT_AiT_Bi_pi(
    q_type const& q,
    Eigen::VectorXd& b) const
{
    auto const N  = q.rows() / 3;
    auto const v1 = this->indices().at(0);
    auto const v2 = this->indices().at(1);
    auto const v3 = this->indices().at(2);
    auto const v4 = this->indices().at(3);

    std::size_t const vi = static_cast<std::size_t>(3u) * v1;
    std::size_t const vj = static_cast<std::size_t>(3u) * v2;
    std::size_t const vk = static_cast<std::size_t>(3u) * v3;
    std::size_t const vl = static_cast<std::size_t>(3u) * v4;

    Eigen::Vector3d const q1 = q.block(vi, 0, 3, 1);
    Eigen::Vector3d const q2 = q.block(vj, 0, 3, 1);
    Eigen::Vector3d const q3 = q.block(vk, 0, 3, 1);
    Eigen::Vector3d const q4 = q.block(vl, 0, 3, 1);

    glm::vec3 o1 = glm::vec3((q1 - q4)(0), (q1 - q4)(1), (q1 - q4)(2));
    glm::vec3 o2 = glm::vec3((q2 - q4)(0), (q2 - q4)(1), (q2 - q4)(2));
    glm::vec3 o3 = glm::vec3((q3 - q4)(0), (q3 - q4)(1), (q3 - q4)(2));

    glm::mat3x3 Dss;
    Dss[0] = o1;
    Dss[1] = o2;
    Dss[2] = o3;
    Dss = glm::transpose(Dss);

    glm::mat3x3 Dm_1;
    Dm_1[0] = glm::vec3(DmInv_(0, 0), DmInv_(0, 1), DmInv_(0, 2));
    Dm_1[1] = glm::vec3(DmInv_(1, 0), DmInv_(1, 1), DmInv_(1, 2));
    Dm_1[2] = glm::vec3(DmInv_(2, 0), DmInv_(2, 1), DmInv_(2, 2));

    glm::mat3x3 f = Dss * Dm_1;

    glm::mat3x3 uy;
    glm::mat3x3 vy;
    glm::mat3x3 ss;
    
    svd_cpu(f[0][0], f[1][0], f[2][0], f[0][1], f[1][1], f[2][1], f[0][2], f[1][2], f[2][2],
        uy[0][0], uy[1][0], uy[2][0], uy[0][1], uy[1][1], uy[2][1], uy[0][2], uy[1][2], uy[2][2],
        ss[0][0], ss[1][0], ss[2][0], ss[0][1], ss[1][1], ss[2][1], ss[0][2], ss[1][2], ss[2][2],
        vy[0][0], vy[1][0], vy[2][0], vy[0][1], vy[1][1], vy[2][1], vy[0][2], vy[1][2], vy[2][2]);
    glm::mat3x3 rr = uy * glm::transpose(vy);
    if (glm::determinant(rr) < 0)
    {
        rr[2] = -rr[2];
    }

    auto const w         = this->wi();
    scalar_type const V0 = std::abs(V0_);
    auto const weight    = w * V0;
    

    glm::mat4x3 DmP;
    DmP[0] = Dm_1[0];
    DmP[1] = Dm_1[1];
    DmP[2] = Dm_1[2];
    glm::vec3 ptt = glm::vec3(-Dm_1[0][0] - Dm_1[1][0] - Dm_1[2][0], -Dm_1[0][1] - Dm_1[1][1] - Dm_1[2][1], -Dm_1[0][2] - Dm_1[1][2] - Dm_1[2][2]);
    DmP[3] = ptt;

    glm::mat3x3 rrr = glm::transpose(rr);
    glm::mat4x3 Dm_1r = rrr * DmP;

    for (int i = 0; i < 4; i++)
    {
        int v = this->indices().at(i) * 3;
        b(v) += weight * Dm_1r[i][0];
        b(v + 1) += weight * Dm_1r[i][1];
        b(v + 2) += weight * Dm_1r[i][2];
    }
}

std::vector<Eigen::Triplet<deformation_gradient_constraint_t::scalar_type>>
deformation_gradient_constraint_t::get_wi_SiT_AiT_Ai_Si(
    positions_type const& p,
    masses_type const& M) const
{
    auto const N  = p.rows();

    auto const w         = this->wi();
    scalar_type const V0 = std::abs(V0_);
    auto const weight    = w * V0;

    std::array<Eigen::Triplet<scalar_type>, 12u * 4u> triplets;
    
    
    Eigen::Vector3d pt;
    pt(0) = -DmInv_(0, 0) - DmInv_(1, 0) - DmInv_(2, 0);
    pt(1) = -DmInv_(0, 1) - DmInv_(1, 1) - DmInv_(2, 1);
    pt(2) = -DmInv_(0, 2) - DmInv_(1, 2) - DmInv_(2, 2);

    glm::mat4x3 DmP;
    DmP[0] = glm::vec3(DmInv_(0, 0), DmInv_(0, 1), DmInv_(0, 2));
    DmP[1] = glm::vec3(DmInv_(1, 0), DmInv_(1, 1), DmInv_(1, 2));
    DmP[2] = glm::vec3(DmInv_(2, 0), DmInv_(2, 1), DmInv_(2, 2));
    glm::vec3 ptt = glm::vec3(-DmInv_(0, 0) - DmInv_(1, 0) - DmInv_(2, 0), -DmInv_(0, 1) - DmInv_(1, 1) - DmInv_(2, 1), -DmInv_(0, 2) - DmInv_(1, 2) - DmInv_(2, 2));
    DmP[3] = ptt;
    glm::mat3x4 DmPT = glm::transpose(DmP);
    glm::mat4x4 st = DmPT * DmP;


    Eigen::MatrixXd DmPlus(3, 4);
    DmPlus.col(0) = DmInv_.row(0);
    DmPlus.col(1) = DmInv_.row(1);
    DmPlus.col(2) = DmInv_.row(2);
    DmPlus.col(3) = pt;
    Eigen::MatrixXd DmPlusT(4,3);
    DmPlusT = DmPlus.transpose();
    Eigen::Matrix4d s = DmPlusT * DmPlus;

    

    for (int i = 0; i < 4; i++)
    {
        int vr = this->indices().at(i) * 3;
        for (int j = 0; j < 4; j++)
        {
            int vc = this->indices().at(j) * 3;
            for (int k = 0; k < 3; k++)
            {
                triplets[i * 12 + j * 3 + k] = { vc + k, vr + k, weight * st[j][i] };
            }
        }
    }
  
    return std::vector<Eigen::Triplet<scalar_type>>{triplets.begin(), triplets.end()};
}

} 