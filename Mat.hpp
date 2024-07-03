#ifndef MAT_HPP
#define MAT_HPP 1
typedef signed char int8_t;

#include <cstdlib>
#include <ctime>
#include <vector>
#include <cassert>
#include <utility>
#include <iostream>
#include <format>

class Mat
{
private:
    size_t m_H;
    size_t m_W;
    std::vector<int8_t> m_data;

public:
    // ctors & dtor:
    Mat(size_t H, size_t W) // constructor use H, W parameters
        : m_H{H}, m_W{W}, m_data(H * W)
    {
        zero();
    }
    Mat(const Mat &other) // copy constructor
        : m_H{other.H()}, m_W{other.W()}, m_data{other.m_data}
    {
    }
    /*Mat(Mat &&other) noexcept // move constructor
        : m_H{other.H()}, m_W{other.W()}, m_data{other.m_data}
    {
        other.m_H = 0;
        other.m_W = 0;
        std::vector<int8_t> empty;
        other.m_data = empty;
    }*/
    Mat &operator=(const Mat &other) // copy assignment operator
    {
        m_H = other.H();
        m_W = other.W();
        m_data = other.m_data;
        return *this;
    }
    /*Mat &operator=(Mat &&other) noexcept // move assignment operator
    {
        if (this != &other)
        {
            m_H = other.H();
            m_W = other.W();
            m_data = other.m_data;
            other.m_H = 0;
            other.m_W = 0;
            std::vector<int8_t> empty;
            other.m_data = empty;
        }
        return *this;
    }*/
    ~Mat() {} // destructor
    // getters:
    size_t H() const
    {
        return m_H;
    }
    size_t W() const
    {
        return m_W;
    }
    int8_t at(size_t i, size_t j) const // an overloading function of two-dimensional .at() which can only read the value
    {
        assert(i < H());
        assert(j < W());
        return m_data[i * W() + j];
    }
    Mat range(size_t i0, size_t i1, size_t j0, size_t j1) const // a select function which return a Mat range from (i0,j0) to (i1, j1)
    {
        assert(i0 < i1 && i1 <= H());
        assert(j0 < j1 && j1 <= W());
        size_t H0 = i1 - i0, W0 = j1 - j0;
        Mat ret = Mat(H0, W0);
        for (size_t i = 0; i < H0; ++i)
        {
            for (size_t j = 0; j < W0; ++j)
            {
                ret.at(i, j) = at(i0 + i, j0 + j);
            }
        }
        return ret;
    }
    // setter:
    int8_t &at(size_t i, size_t j) // two-dimensional .at() which can return and change the corresponding value of Mat[i][j]
    {
        assert(i < H());
        assert(j < W());
        return m_data[i * W() + j];
    }
    // fillers:
    void zero() // fill the Mat with 0s
    {
        std::fill(std::begin(m_data), std::end(m_data), 0);
    }
    void M_rand() // fill the Mat with random numbers
    {
        srand(time(0) + ::rand());
        for (size_t i = 0; i < H(); ++i)
        {
            for (size_t j = 0; j < W(); ++j)
            {
                at(i, j) = ::rand();
            }
        }
    }
    // operator ==
    bool operator==(const Mat &rhs) const
    {
        if (H() != rhs.H() || W() != rhs.W())
            return false;
        for (size_t i = 0; i < H(); ++i)
        {
            for (size_t j = 0; j < W(); ++j)
            {
                if (at(i, j) != rhs.at(i, j))
                    return false;
            }
        }
        return true;
    }
    // subMat returns a H0*W0 matrix resulted from divided H*W matrix
    Mat subMat(size_t h1, size_t w1, size_t H0, size_t W0) const
    {
        assert(this->H() % H0 == 0);
        assert(this->W() % W0 == 0);
        assert((h1 + 1) * H0 <= this->H()); // H1*H0 = H(); H0 is a global constant; h1 is an outside iterator
        assert((w1 + 1) * W0 <= this->W()); // W1*W0 = W(); W0 is a global constant; w1 is an outside iterator
        auto ret = Mat(H0, W0);
        for (size_t i = 0; i < H0; ++i)
        {
            for (size_t j = 0; j < W0; ++j)
            {
                ret.at(i, j) = this->at(h1 * H0 + i, w1 * W0 + j);
            }
        }
        return ret;
    };
    // subMatAdd adds a H0*W0 submatrix to a H*W matrix begin at (h1*H0, w1*W0)
    void subMatAdd(size_t h1, size_t w1, const Mat &sub)
    {
        auto H0 = sub.H(), W0 = sub.W(); // sub is a H0*W0 matrix
        assert(this->H() % H0 == 0);
        assert(this->W() % W0 == 0);
        assert((h1 + 1) * H0 <= this->H()); // H1*H0 = H(); H0 is a global constant; h1 is an outside iterator
        assert((w1 + 1) * W0 <= this->W()); // W1*W0 = W(); W0 is a global constant; w1 is an outside iterator
        for (size_t i = 0; i < H0; ++i)
        {
            for (size_t j = 0; j < W0; ++j)
            {
                this->at(h1 * H0 + i, w1 * W0 + j) += sub.at(i, j);
            }
        }
    }
    // pad *this with 0s in all four directions
    Mat pad(size_t upper_pad, size_t lower_pad, size_t left_pad, size_t right_pad) const
    {
        auto ret = Mat(H() + upper_pad + lower_pad, W() + left_pad + right_pad);
        // original part
        for (size_t i = 0; i < H(); ++i)
        {
            for (size_t j = 0; j < W(); ++j)
            {
                ret.at(i + upper_pad, j + left_pad) = this->at(i, j);
            }
        }
        return ret;
    }
    // original part is placed up-left, while down-right are still ret's initialized 0s
    Mat padSubMat(size_t H0, size_t W0) const
    {
        size_t new_H = (this->H() + H0 - 1) / H0 * H0;
        size_t new_W = (this->W() + W0 - 1) / W0 * W0;
        return pad(0, new_H - H(), 0, new_W - W());
    }
    // display the matrix
    void display() const
    {
        std::cerr << std::format("H: {}, W: {}", H(), W()) << "\n";
        std::cerr << "      ";
        for (size_t i = 0; i < this->W(); ++i)
        {
            std::cerr << std::format("{:5}", i);
        }
        std::cerr << "\n";
        std::cerr << "      ";
        for (size_t i = 0; i < this->W(); ++i)
        {
            std::cerr << "  ---";
        }
        std::cerr << "\n";
        for (size_t i = 0; i < this->H(); ++i)
        {
            std::cerr << std::format("{:5}|", i);
            for (size_t j = 0; j < this->W(); ++j)
            {
                std::cerr << std::format("{:5}", at(i, j));
            }
            std::cerr << "\n";
        }
    }
};

// golden matmul
Mat golden_matmul(const Mat &lhm, const Mat &rhm)
{
    assert(lhm.W() == rhm.H());
    const size_t M = lhm.H(), K = lhm.W(), N = rhm.W();

    auto ret = Mat(M, N);

    for (size_t i = 0; i < M; ++i)
    {
        for (size_t j = 0; j < N; ++j)
        {
            for (size_t k = 0; k < K; ++k)
            {
                ret.at(i, j) += lhm.at(i, k) * rhm.at(k, j);
            }
        }
    }
    return ret;
};

// golden conv2d with different strides
Mat golden_Matconv(const Mat &fea, const Mat &kel, const int &stride)
{
    assert(kel.H() % 2 == 1); // kernel should be a square matrix whose sidelength should be an odd number
    assert(kel.W() % 2 == 1);
    size_t retH = (fea.H() - kel.H() + kel.H() / 2 * 2) / stride + 1;
    size_t retW = (fea.W() - kel.W() + kel.W() / 2 * 2) / stride + 1;
    auto ret = Mat(retH, retW);
    auto padded_fea = fea.pad(kel.H() / 2, kel.H() / 2, kel.W() / 2, kel.W() / 2);
    for (size_t i = 0; i < retH; ++i)
    {
        for (size_t j = 0; j < retW; ++j)
        {
            for (size_t k = 0; k < kel.H(); ++k)
            {
                for (size_t l = 0; l < kel.W(); ++l)
                {
                    ret.at(i, j) += padded_fea.at(stride * i + k, stride * j + l) * kel.at(k, l);
                }
            }
        }
    }
    return ret;
};

// matmul using img2col and datalayout conversion
Mat matmul(const Mat &lhm, const Mat &rhm, size_t M0, size_t K0, size_t N0)
{
    assert(lhm.W() == rhm.H());
    auto padded_fea = lhm.padSubMat(M0, K0);
    auto padded_kel = rhm.padSubMat(K0, N0);

    auto M1 = padded_fea.H() / M0;
    auto K1 = padded_fea.W() / K0;
    auto N1 = padded_kel.W() / N0;
    auto psb = Mat(padded_fea.H(), padded_kel.W());

    // 1. L2::MK->L1::M1K1M0K0 + 3. L1::M1K1M0K0->LMB::M1K1M0K0
    auto divided_fea = std::vector<Mat>(M1 * K1, Mat(M0, K0));
    for (size_t i = 0; i < M1; ++i)
    {
        for (size_t j = 0; j < K1; ++j)
        {
            divided_fea[i * K1 + j] = padded_fea.subMat(i, j, M0, K0);
        }
    }

    // 2. L2::KN->L1::N1K1K0N0 + 4. L1::N1K1K0N0->RMB::N1K1K0N0
    auto divided_kel = std::vector<Mat>(K1 * N1, Mat(K0, N0));
    for (size_t i = 0; i < K1; ++i)
    {
        for (size_t j = 0; j < N1; ++j)
        {
            divided_kel[i * N1 + j] = padded_kel.subMat(i, j, K0, N0);
        }
    }

    // 5. LMB::M1K1M0K0+RMB::N1K1K0N0->PSB::M1N1M0N0
    for (size_t i = 0; i < M1; ++i) // i for iterator m1
    {

        for (size_t j = 0; j < N1; ++j) // j for iterator n1
        {

            for (size_t k = 0; k < K1; ++k) // add all partial sum matrixes up
            {
                const auto &L_sub = divided_fea[i * K1 + k];
                const auto &R_sub = divided_kel[k * N1 + j];
                auto sub_res = golden_matmul(L_sub, R_sub);
                psb.subMatAdd(i, j, sub_res);
            }
        }
    }

    // 6. PSB::M1N1M0N0->L1::M1N1M0N0 / 7. PSB::M1N1M0N0->L2::MN + 8. L1::M1N1M0N0->L2::MN
    return psb.range(0, lhm.H(), 0, rhm.W());
};

#endif // MAT_HPP