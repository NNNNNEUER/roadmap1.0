#ifndef RDTU_HPP
#define RDTU_HPP

#include "DLU.hpp"
#include "Mat.hpp"

class RDTU
{
public:
    RDTU(DLU &DLU, size_t size_cacheline) : L1_cache(DLU), m_RMB(size_cacheline) {}
    // set Weight
    void configureWeight(size_t L1_addr, size_t Co0, size_t Kh, size_t Kw)
    {
        m_L1_addr = L1_addr;
        m_Co0 = Co0;
        m_Kh = Kh;
        m_Kw = Kw;
    }
    // ���ص�RMB����Ϊ(Co0,Kh*Kw)�������ﱾ����û��data layout�任
    void load()
    {
        // �����Ͼ��Ǵ�L1��copy��L1_addr��ʼ��
        // [0,m_Co0*m_Kw*Kh)������Ҫѭ������Ϊ�˽ӿڳ�����ʱ����
        // �ṹ��Co0 * (Kw*Kh)�ľ���,ÿ��Ԫ��һ��cacheline
#define OFFSETRDTU(oi, ki, kj) (oi * m_Kh * m_Kw + ki * m_Kw + kj)
        for (size_t oi = 0; oi < m_Co0; ++oi)
        {
            for (size_t ki = 0; ki < m_Kh; ++ki)
            {
                for (size_t kj = 0; kj < m_Kw; ++kj)
                {
                    m_RMB[OFFSETRDTU(oi, ki, kj)] = L1_cache.getWeightCacheLine(m_L1_addr, oi, ki, kj);
                }
            }
        }
    }

    // ����(Co0,Kh*Kw)�����е�(Co0,Kh*Kw) cacheline
    CubeCacheLine getCacheLine(size_t i, size_t j)
    {
        assert(i < (m_Co0) && j < (m_Kh * m_Kw));
        return m_RMB[i * (m_Kh * m_Kw) + j];
    }

    void displayMat(size_t channel)
    {
        Mat mat(m_Co0, m_Kh * m_Kw);
        for (size_t i = 0; i < m_Co0; ++i)
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
    size_t m_Co0;
    size_t m_Kh;
    size_t m_Kw;
    const size_t m_Ci0 = C0;
    DLU &L1_cache;
    CubeCache m_RMB;
};

#endif // RDTU_HPP