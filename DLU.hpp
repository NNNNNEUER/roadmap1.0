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
    // 加载以(H1,W1,C1)子块为中心，大小为(H0+Kh-1,W0+Kw-1)的subFeature到addr(上下左右Kh/2,
    // Kw/2的padding,越界为0)，存到L1中
    void loadSubPaddedFeature(size_t addr, size_t H1, size_t W1, size_t C1)
    {
        // Reminder:subFeature在L1中是一个(H0+Kh-1,W0+Kw-1)矩阵，每个元素是一个cache
        // line，包含C0个channels.
        // 遍历所有要加载的元素i <- [0, H0+Kh-1), j <- [0, W0+Kw-1)
        for (size_t i = 0; i < m_H0 + m_Kh - 1; i++)
        {
            for (size_t j = 0; j < m_W0 + m_Kw - 1; j++)
            {
                // 计算当前元素相对于L1的addr的offset
                // 采用了row major的存储方式
                size_t L1_offset = i * (m_W0 + m_Kw - 1) + j;

                // 从L2中读取元素，暂时一个一个元素加载 TODO::优化为一次加载多个元素
                for (size_t c = 0; c < m_C0; c++)
                {
                    // 计算当前元素在L2中的地址
                    size_t L2_addr_row = H1 * m_H0 + (i - m_Kh / 2);
                    size_t L2_addr_col = W1 * m_W0 + (j - m_Kw / 2);
                    size_t L2_addr_cha = C1 * m_C0 + c;
                    // 从L2中读取元素
                    // judge whether (im_row, im_col, im_cha) is out of bound
                    if (L2_addr_row < m_H && L2_addr_col < m_W && L2_addr_cha < m_C)
                    {
                        size_t L2_addr_offset = L2_addr_row * m_W * m_C + L2_addr_col * m_C + L2_addr_cha;
                        m_L1_cache[addr + L1_offset][c] = L2_cache.read(m_Feature_L2_addr + L2_addr_offset);
                    }
                    else
                    {
                        m_L1_cache[addr + L1_offset][c] = 0;
                    }
                }
            }
        }
    }
    // 加载(Co1,Kh,Kw,Ci1)子块到addr，存到L1中
    void loadSubWeight(size_t addr, size_t Co1, size_t Ci1)
    {
        // Reminder:subWeight在L1中是一个(Co0,Kh,Kw)矩阵，每个元素是一个cache
        // line，包含Ci0个channels.
        // 遍历所有要加载的元素
        // Co0个Co channels: oi <- [0, Co0)
        for (size_t oi = 0; oi < m_Co0; oi++)
        {
            // 遍历weight的每个元素 ki <- [0, Kh), kj <- [0, Kw)
            for (size_t ki = 0; ki < m_Kh; ki++)
            {
                for (size_t kj = 0; kj < m_Kw; kj++)
                {
                    // 计算当前元素相对于L1的addr的offset
                    // 采用了row major的存储方式
                    size_t L1_offset = oi * m_Kh * m_Kw + ki * m_Kw + kj;

                    // 从L2中读取元素，暂时Ci0个元素逐个加载 TODO::优化为一次加载多个元素
                    for (size_t ci = 0; ci < m_Ci0; ci++)
                    {
                        // 计算当前元素在L2中的地址
                        size_t L2_addr_cho = Co1 * m_Co0 + oi;
                        size_t L2_addr_row = ki;
                        size_t L2_addr_col = kj;
                        size_t L2_addr_chi = Ci1 * m_Ci0 + ci;
                        // 从L2中读取元素
                        // judge whether (im_cho, im_chi) is out of bound
                        if (L2_addr_cho < m_Co && L2_addr_chi < m_Ci)
                        {
                            size_t L2_addr_offset = L2_addr_cho * m_Kh * m_Kw * m_Ci + L2_addr_row * m_Kw * m_Ci + L2_addr_col * m_Ci + L2_addr_chi;
                            m_L1_cache[addr + L1_offset][ci] = L2_cache.read(m_Weight_L2_addr + L2_addr_offset);
                        }
                        else
                        {
                            m_L1_cache[addr + L1_offset][ci] = 0;
                        }
                    }
                }
            }
        }
    }
    // 把从addr开始的内存interpret为一个(H0+Kh-1,W0+Kw-1)的Feature矩阵，每个元素为一个cacheline(C0
    // elements)
    CubeCacheLine getFeatureCacheLine(size_t addr, size_t i, size_t j)
    {
        return m_L1_cache[addr + i * (m_W0 + m_Kw - 1) + j];
    }
    // 把从addr开始的内存interpret为一个(Co0,Kh,Kw)的Weight矩阵，每个元素为一个cacheline(Ci0
    // elements)
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