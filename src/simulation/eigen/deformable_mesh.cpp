#include "deformable_mesh.h"

#include "deformation_gradient_constraint.h"
#include "positional_constraint.h"

#include <array>

namespace pd {

void deformable_mesh_t::add_positional_constraint(int vi, scalar_type wi)
{
    auto const& positions = this->p0();
    this->constraints().push_back(std::make_unique<positional_constraint_t>(
        std::initializer_list<std::uint32_t>{static_cast<std::uint32_t>(vi)},
        wi,
        positions));
}

void deformable_mesh_t::constrain_deformation_gradient(scalar_type wi)
{
    auto const& positions = this->p0();
    auto const& elements  = this->elements();

    for (auto i = 0u; i < elements.rows(); ++i)
    {
        auto const element = elements.row(i);
        auto constraint    = std::make_unique<deformation_gradient_constraint_t>(
            std::initializer_list<std::uint32_t>{
                static_cast<std::uint32_t>(element(0)),
                static_cast<std::uint32_t>(element(1)),
                static_cast<std::uint32_t>(element(2)),
                static_cast<std::uint32_t>(element(3))},
            wi,
            positions);

        this->constraints().push_back(std::move(constraint));
    }
}

} // namespace pd