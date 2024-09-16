#include <energy/elasticity.h>

template<typename Scalar>
inline ElasticEnergy<Scalar>::ElasticEnergy(int& hessianIdxOffset) :
    Energy<Scalar>(hessianIdxOffset)
{
}


template<typename Scalar>
inline Energy<Scalar>::Energy(int hessianIdxOffset) :hessianIdxOffset(hessianIdxOffset)
{
}

template<typename Scalar>
int Energy<Scalar>::NNZ() const
{
    return nnz;
}

template<typename Scalar>
inline void Energy<Scalar>::SetHessianPtr(Scalar* hessianVal, int* hessianRowIdx, int* hessianColIdx)
{
    if (hessianIdxOffset == -1)return;
    this->hessianVal = hessianVal + hessianIdxOffset;
    this->hessianRowIdx = hessianRowIdx + hessianIdxOffset;
    this->hessianColIdx = hessianColIdx + hessianIdxOffset;
}

template class Energy<float>;
template class Energy<double>;

template class ElasticEnergy<float>;
template class ElasticEnergy<double>;