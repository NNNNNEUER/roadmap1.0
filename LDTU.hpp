#ifndef LDTU_HPP
#define LDTU_HPP

#include <cstddef>
#include "CubeCache.hpp"
#include "DLU.hpp"
#include "Mat.hpp"

class LDTU
{
public:
    LDTU(DLU &DLU, size_t size_cacheline) : L1_cache(DLU), m_LMB(size_cacheline){};
    // set Feature in L1
    void configureFeature(size_t L1_addr, size_t H0, size_t W0, size_t Kh, size_t Kw)
    {
        m_L1_addr = L1_addr;
        m_H0 = H0;
        m_W0 = W0;
        m_Kh = Kh;
        m_Kw = Kw;
    }
    // 对L1中的(H0+Kh-1,W0+Kw-1)的subPaddedFeature做im2col到(H0*W0,Kh*Kw)的矩阵
    void load_im2col()
    {
#define OFFSETLDTU(i, j, ki, kj) ((i * m_W0 + j) * (m_Kh * m_Kw) + (ki * m_Kw + kj))
        // row
        for (size_t i = 0; i < m_H0; i++)
        {
            for (size_t j = 0; j < m_W0; j++)
            {
                // col
                for (size_t ki = 0; ki < m_Kh; ki++)
                {
                    for (size_t kj = 0; kj < m_Kw; kj++)
                    {
                        m_LMB[OFFSETLDTU(i, j, ki, kj)] = L1_cache.getFeatureCacheLine(m_L1_addr, i + ki, j + kj);
                    }
                }
            }
        }
    }
    // 获得(H0*W0,Kh*Kw)的矩阵的第i行第j列的元素(CubeCacheLine)
    CubeCacheLine getCacheLine(size_t i, size_t j)
    {
        assert(in_bound(i, j));
        return m_LMB[i * m_Kh * m_Kw + j];
    }

    void displayMat(size_t channel)
    {
        Mat mat(m_H0 * m_W0, m_Kh * m_Kw);
        for (size_t i = 0; i < m_H0 * m_W0; ++i)
        {
            for (size_t j = 0; j < m_Kh * m_Kw; ++j)
            {
                mat.at(i, j) = getCacheLine(i, j)[channel];
            }
        }
        mat.display();
    }

private:
    size_t m_L1_addr;
    size_t m_H0;
    size_t m_W0;
    const size_t m_C0 = C0;
    size_t m_Kh;
    size_t m_Kw;
    DLU &L1_cache;
    CubeCache m_LMB;
    bool in_bound(size_t i, size_t j)
    {
        return (i < m_H0 * m_W0) && (j < m_Kh * m_Kw);
    }
};

#endif // LDTU_HPP