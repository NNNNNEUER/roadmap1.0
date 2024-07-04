#ifndef CU_HPP
#define CU_HPP

#include "CubeCache.hpp"
#include "LDTU.hpp"
#include "RDTU.hpp"

class CU
{
public:
    CU(LDTU &LDTU, RDTU &RDTU, size_t size_cacheline) : LMB(LDTU), RMB(RDTU), m_PSB(size_cacheline) {}
    // Set two matrix formats, note that M=H0*W0, K=Kh*Kw, N=Co0
    void configureMat(size_t M, size_t K, size_t N)
    {
        m_M = M;
        m_K = K;
        m_N = N;
    }
    // Multiply submatrix, accumulate the result in PSB
    void matmul(size_t M1, size_t K1, size_t N1)
    {
        // M : H0 * W0
        // K : Kh * Kw
        // N : Co0

        // LMB : (H0 * W0) * (Kh * Kw) = M * K = (M1 * M0) * (K1 * K0)
        // RMB : (Co0) * (Kh * Kw) = N * K = N * (K1 * K0)
        // PSB : (H0 * W0) * N0 This is still HWC,
        // but note that N0 = C0 = 16,
        // the last dimension has 16 channels to fill up a cacheline,
        // if Co0 < C0 (N < N0) then there will be padding

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

        // RDTU holds the transpose of the right matrix
        std::vector<CubeCacheLine> RMB_line(m_N0);
        for (size_t i = 0; i < m_N0; i++)
        {
            auto N_i = N1 * m_N0 + i;
            if (N_i < m_N)
            {
                RMB_line[i] = RMB.getCacheLine(N_i, K1);
            }
        }

        // Note that MatMul requires right to be transposed
        std::vector<CubeCacheLine> result = MatMul(LMB_line, RMB_line);
        // PSB: (H0 * W0) * Co0
        for (size_t i = 0; i < m_M0; i++)
        {
            auto M_i = M1 * m_M0 + i;
            // m_N0 = C0
            // There are CEIL(m_N/m_N0) = ((m_N + m_n0-1) /m_N0) cachelines in a row,
            // storing Co0 channels (if Co0 < C0 (N < N0) then there will be padding)
            m_PSB[M_i * ((m_N + m_N0 - 1) / m_N0) + N1].bitwiseAddCacheLine(result[i]);
        }
    }
    // clear PSB
    void
    clearPSB()
    {
        for (size_t i = 0; i < m_PSB.size(); i++)
        {
            m_PSB[i].clear();
        }
    }
    // inquire PSB
    CubeCache &getPSB()
    {
        return m_PSB;
    }

private:
    size_t m_M;
    size_t m_K;
    size_t m_N;
    const size_t m_M0 = C0; // since
    size_t m_K0 = 15;       // K0 can be any number, doesn't matter
    const size_t m_N0 = C0; // to use cacheline as the container, N0 = C0 = 16
    LDTU &LMB;
    RDTU &RMB;
    CubeCache m_PSB;
};

#endif // CU_HPP