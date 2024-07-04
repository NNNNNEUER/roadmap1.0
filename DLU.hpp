#ifndef DLU_HPP
#define DLU_HPP

#include <cstddef>
#include "L2.hpp"
#include "CubeCache.hpp"

class DLU
{
public:
    DLU(L2 &L2, size_t size_cacheline) : L2_cache(L2), m_L1_cache(size_cacheline) {}

    // setter for feature
    void configureFeature(size_t L2_addr, size_t H, size_t W, size_t C, size_t H0, size_t W0)
    {
        m_H = H;
        m_W = W;
        m_C = C;
        m_H0 = H0;
        m_W0 = W0;
        // m_C0 is fixed
        m_Feature_L2_addr = L2_addr;
    }
    // setter for weight
    void configureWeight(size_t L2_addr, size_t Co, size_t Kh, size_t Kw, size_t Ci, size_t Co0)
    {
        m_Co = Co;
        m_Kh = Kh;
        m_Kw = Kw;
        m_Ci = Ci;
        // m_Ci0 is fixed
        m_Co0 = Co0;
        m_Weight_L2_addr = L2_addr;
    }
    // Load subfeatures of size (H0+kh-1, W0+Kw-1) centered at subblocks (H1,W1,C1)
    // to addr (Kh/2,Kw/2 padding in all four directions, out of bounds is 0), stored in L1
    void loadSubPaddedFeature(size_t addr, size_t H1, size_t W1, size_t C1)
    {
        // Reminder: A subFeature in L1 is a (H0+Kh-1,W0+Kw-1) matrix,
        // where each element is a cacheline with C0 channels.
        // Iterate over all elements to load i <- [0, H0+Kh-1), j <- [0, W0+Kw-1)
        for (size_t i = 0; i < m_H0 + m_Kh - 1; i++)
        {
            for (size_t j = 0; j < m_W0 + m_Kw - 1; j++)
            {
// Calculate the offset of the current element's addr relative to L1
// row major storage is used
#define OFFSETDLUFeature(i, j) (i * (m_W0 + m_Kw - 1) + j)
                // Read elements from L2, temporarily load them one by one
                for (size_t c = 0; c < m_C0; c++)
                {
                    // Calculate the address of the current element in L2
                    size_t L2_addr_row = H1 * m_H0 + (i - m_Kh / 2);
                    size_t L2_addr_col = W1 * m_W0 + (j - m_Kw / 2);
                    size_t L2_addr_cha = C1 * m_C0 + c;
                    // Read elements from L2
                    // judge whether (im_row, im_col, im_cha) is out of bound
                    if (L2_addr_row < m_H && L2_addr_col < m_W && L2_addr_cha < m_C)
                    {
                        size_t L2_addr_offset =
                            L2_addr_row * m_W * m_C +
                            L2_addr_col * m_C +
                            L2_addr_cha;
                        m_L1_cache[addr + OFFSETDLUFeature(i, j)][c] =
                            L2_cache.read(m_Feature_L2_addr + L2_addr_offset);
                    }
                    else
                    {
                        m_L1_cache[addr + OFFSETDLUFeature(i, j)][c] = 0;
                    }
                }
            }
        }
    }
    // Load subweight of size (Co0,Kh,Kw) centered at subblocks (Co1,Ci1) into addr and store in L1
    void loadSubWeight(size_t addr, size_t Co1, size_t Ci1)
    {
        // Reminder:subWeight is a (Co0,Kh,Kw) matrix in L1,
        // where each element is a cacheline, containing Ci0 channels.
        // Iterate over all the elements to load
        // Co0 Co channels: oi <- [0, Co0)
        for (size_t oi = 0; oi < m_Co0; oi++)
        {
            // Iterate over each element of weight ki <- [0, Kh), kj <- [0, Kw)
            for (size_t ki = 0; ki < m_Kh; ki++)
            {
                for (size_t kj = 0; kj < m_Kw; kj++)
                {
// Calculate the offset of the current element's addr relative to L1
// row major storage is used
#define OFFSETDLUWeight(oi, ki, kj) (oi * m_Kh * m_Kw + ki * m_Kw + kj)
                    // Load elements from L2, temporarily Ci0 elements one by one
                    for (size_t ci = 0; ci < m_Ci0; ci++)
                    {
                        // Calculate the address of the current element in L2
                        size_t L2_addr_cho = Co1 * m_Co0 + oi;
                        size_t L2_addr_row = ki;
                        size_t L2_addr_col = kj;
                        size_t L2_addr_chi = Ci1 * m_Ci0 + ci;
                        // Read elements from L2
                        // judge whether (im_cho, im_chi) is out of bound
                        if (L2_addr_cho < m_Co && L2_addr_chi < m_Ci)
                        {
                            size_t L2_addr_offset =
                                L2_addr_cho * m_Kh * m_Kw * m_Ci +
                                L2_addr_row * m_Kw * m_Ci +
                                L2_addr_col * m_Ci +
                                L2_addr_chi;
                            m_L1_cache[addr + OFFSETDLUWeight(oi, ki, kj)][ci] =
                                L2_cache.read(m_Weight_L2_addr + L2_addr_offset);
                        }
                        else
                        {
                            m_L1_cache[addr + OFFSETDLUWeight(oi, ki, kj)][ci] = 0;
                        }
                    }
                }
            }
        }
    }
    // interpret the memory from addr as a (H0+Kh-1,W0+Kw-1) Feature matrix
    // with each entry being a cacheline(C0 elements)
    CubeCacheLine getFeatureCacheLine(size_t addr, size_t i, size_t j)
    {
        return m_L1_cache[addr + i * (m_W0 + m_Kw - 1) + j];
    }
    // interpret the memory from addr as a (Co0,Kh,Kw) Weight matrix
    // with each entry a cacheline(Ci0 elements)
    CubeCacheLine getWeightCacheLine(size_t addr, size_t oi, size_t ki, size_t kj)
    {
        return m_L1_cache[addr + oi * m_Kh * m_Kw + ki * m_Kw + kj];
    }

private:
    size_t m_H;
    size_t m_W;
    size_t m_C;
    size_t m_H0;
    size_t m_W0;
    const size_t m_C0 = C0;
    size_t m_Feature_L2_addr;

    size_t m_Co;
    size_t m_Kh;
    size_t m_Kw;
    size_t m_Ci;
    const size_t m_Ci0 = C0;
    size_t m_Co0;
    size_t m_Weight_L2_addr;

    L2 &L2_cache;
    CubeCache m_L1_cache;
};

#endif // DLU_HPP