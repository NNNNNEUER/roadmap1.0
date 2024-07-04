#ifndef CUBECACHE_HPP
#define CUBECACHE_HPP

#include <cstdint>
#include <cstddef>
#include <cassert>
#include <iostream>
#include <vector>

constexpr size_t C0 = 16;

// CubeCacheLine is the fundamental unit of L1, LMB, RMB, MMB and PSB
class CubeCacheLine
{
public:
    CubeCacheLine() : m_data(C0) {} // CubeCacheLine is a int_8 vector of size 16

    int8_t operator[](size_t index) const { return m_data.at(index); }
    int8_t &operator[](size_t index) { return m_data.at(index); }

    void clear() { std::fill(m_data.begin(), m_data.end(), 0); }
    void bitwiseAddCacheLine(const CubeCacheLine &other)
    {
        for (size_t i = 0; i < C0; i++)
        {
            m_data[i] += other[i];
        }
    }
    void print() const
    {
        for (size_t i = 0; i < C0; i++)
        {
            std::cout << (int)m_data[i] << " ";
        }
        std::cout << std::endl;
    }

private:
    std::vector<int8_t> m_data;
};

// L1, LMB, RMB, MMB and PSB are all CubeCache
class CubeCache
{
public:
    CubeCache(size_t size) : m_size(size), m_data(size) {}
    size_t size() const { return m_size; }

    CubeCacheLine operator[](size_t index) const { return m_data.at(index); }
    CubeCacheLine &operator[](size_t index) { return m_data.at(index); }

private:
    size_t m_size;
    std::vector<CubeCacheLine> m_data;
};

// lhs <- LMB,是im2col后的feature,
// rhs <- RMB,是weight的转置
std::vector<CubeCacheLine> MatMul(std::vector<CubeCacheLine> &lhs, std::vector<CubeCacheLine> &rhs)
{
    assert(lhs.size() == C0);
    assert(rhs.size() == C0);
    std::vector<CubeCacheLine> result(C0);
    for (size_t i = 0; i < C0; i++)
    {
        for (size_t j = 0; j < C0; j++)
        {
            for (size_t k = 0; k < C0; k++)
            {
                result[i][j] += lhs[i][k] * rhs[j][k]; // 注意 right[**j**][**k**]
            }
        }
    }
    return result;
}

#endif // CUBECACHE_HPP