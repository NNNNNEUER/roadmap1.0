#ifndef CU_HPP
#define CU_HPP

#include "CubeCache.hpp"
#include "LDTU.hpp"
#include "RDTU.hpp"

class CU
{
public:
    CU(LDTU &LDTU, RDTU &RDTU, size_t size_cacheline) : LMB(LDTU), RMB(RDTU), m_PSB(size_cacheline) {}
    // �������������ʽ,ע��M=H0*W0, K=Kh*Kw(û��*16), N=Co0
    void configureMat(size_t M, size_t K, size_t N)
    {
        m_M = M;
        m_K = K;
        m_N = N;
    }
    // ���Ӿ�����PSB���ۼ� TODO::��������Ҿ��������⣬�Ȼ�������������Բ���
    void matmul(size_t M1, size_t K1, size_t N1)
    {
        // M : H0 * W0
        // K : Kh * Kw
        // N : Co0

        // LMB : (H0 * W0) * (Kh * Kw) = M * K = (M1 * M0) * (K1 * K0)
        // RMB : (Co0) * (Kh * Kw) = N * K = N * (K1 * K0)
        // PSB : (H0 * W0) * Co0, ��������HWC����ע�����һ��ά��16��channelsһ��cacheline�����Ի���padding����

        // LDTU
        std::vector<CubeCacheLine> LMB_line(m_M0);
        for (size_t i = 0; i < m_M0; i++)
        {
            auto M_i = M1 * m_M0 + i;
            if (M_i < m_M)
            {
                LMB_line[i] = LMB.getCacheLine(M_i, K1);
            }
        }

        // RDTU �д�����Ҿ����ת��
        std::vector<CubeCacheLine> RMB_line(m_N0);
        for (size_t i = 0; i < m_N0; i++)
        {
            auto N_i = N1 * m_N0 + i;
            if (N_i < m_N)
            {
                RMB_line[i] = RMB.getCacheLine(N_i, K1);
            }
        }

        // ע��MatMulҪ��right��ת�ú��
        std::vector<CubeCacheLine> result = MatMul(LMB_line, RMB_line);
        // PSB: (H0 * W0) * Co0
        for (size_t i = 0; i < m_M0; i++)
        {
            auto M_i = M1 * m_M0 + i;
            // m_N0 = Co0, һ����CEIL(m_N/m_N0) = ((m_N + m_N0 - 1) / m_N0)��cacheline���洢Co0��channels
            m_PSB[M_i * ((m_N + m_N0 - 1) / m_N0) + N1].addCacheline(result[i]);
        }
    }
    // ���PSB
    void
    clearPSB()
    {
        for (size_t i = 0; i < m_PSB.size(); i++)
        {
            m_PSB[i].clear();
        }
    }
    // ����PSB
    CubeCache &getPSB()
    {
        return m_PSB;
    }

private:
    size_t m_M;
    size_t m_K;
    size_t m_N;
    const size_t m_M0 = 16;
    const size_t m_K0 = 16;
    const size_t m_N0 = 16;
    LDTU &LMB;
    RDTU &RMB;
    CubeCache m_PSB;
};

#endif // CU_HPP