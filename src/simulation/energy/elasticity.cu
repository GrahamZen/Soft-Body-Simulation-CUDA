#include <energy/elasticity.h>

template<typename HighP>
inline ElasticEnergy<HighP>::ElasticEnergy(int& hessianIdxOffset) :
    Energy<HighP>(hessianIdxOffset)
{
}


template<typename HighP>
inline Energy<HighP>::Energy(int hessianIdxOffset) :hessianIdxOffset(hessianIdxOffset)
{
}

template<typename HighP>
inline void Energy<HighP>::SetHessianPtr(HighP* hessianVal, int* hessianRowIdx, int* hessianColIdx)
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