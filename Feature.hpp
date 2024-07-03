#ifndef FEATURE_HPP
#define FEATURE_HPP 1

#include "Mat.hpp"
#include "Weight.hpp"
#include <ctime>
#include <cassert>
#include <cstring>
#include <format>
#include <iostream>
#include <vector>
#include <algorithm>

class Feature
{
private:
    size_t f_H;
    size_t f_W;
    size_t f_C;
    std::vector<int8_t> f_data;

public:
    // ctors & dtor:
    Feature(size_t H, size_t W, size_t C) // constructor use H, W, C parameters
        : f_H{H}, f_W{W}, f_C{C}, f_data(H * W * C)
    {
        std::fill(std::begin(f_data), std::end(f_data), 0);
    }
    Feature(const Feature &other) // copy constructor
        : f_H{other.H()}, f_W{other.W()}, f_C{other.C()}, f_data{other.f_data}
    {
    }
    /*Feature(Feature &&other) noexcept // move constructor
        : f_H{other.H()}, f_W{other.W()}, f_C{other.C()}, f_data{other.f_data}
    {
        other.f_H = 0;
        other.f_W = 0;
        other.f_C = 0;
        std::vector<int8_t> empty;
        other.f_data = empty;
    }*/
    Feature &operator=(const Feature &other) // copy assignment operator
    {
        f_H = other.H();
        f_W = other.W();
        f_C = other.C();
        f_data = other.f_data;
        return *this;
    }
    /*Feature &operator=(Feature &&other) noexcept // move assignment operator
    {
        if (this != &other)
        {
            f_H = other.H();
            f_W = other.W();
            f_C = other.C();
            f_data = other.f_data;
            other.f_H = 0;
            other.f_W = 0;
            other.f_C = 0;
            std::vector<int8_t> empty;
            other.f_data = empty;
        }
        return *this;
    }*/
    ~Feature() {} // destructor
    // getters:
    size_t H() const
    {
        return f_H;
    }
    size_t W() const
    {
        return f_W;
    }
    size_t C() const
    {
        return f_C;
    }
    int8_t at(size_t i, size_t j, size_t k) const // an overloading function of two-dimensional .at() which can only read the value
    {
        assert(i < H());
        assert(j < W());
        assert(k < C());
        return f_data[i * W() * C() + j * C() + k];
    }
    // setter:
    int8_t &at(size_t i, size_t j, size_t k) // three-dimensional .at() which can return and change the corresponding value of Feature[i][j][k]
    {
        assert(i < H());
        assert(j < W());
        assert(k < C());
        return f_data[i * W() * C() + j * C() + k];
    }
    // fillers:
    void F_rand() // fill the Feature with random int8_t
    {
        srand(time(0) + ::rand());
        for (size_t i = 0; i < H(); ++i)
        {
            for (size_t j = 0; j < W(); ++j)
            {
                for (size_t k = 0; k < C(); ++k)
                {
                    at(i, j, k) = ::rand();
                }
            }
        }
    }
    // make up zeros
    int8_t atElseZero(size_t i, size_t j, size_t k) const // return 0 if the index is out of range
    {
        if (i < H() && j < W() && k < C())
            return f_data[i * W() * C() + j * C() + k];
        else
            return 0;
    }
    // display Feature
    void display(size_t C) const
    {
        assert(C < this->C());
        std::cerr << std::format("H: {}, W: {}", H(), W()) << "\n";
        std::cerr << std::format("Channel: {}", C) << "\n";
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
                std::cerr << std::format("{:5}", at(i, j, C));
            }
            std::cerr << "\n";
        }
    }
    // operator ==
    bool operator==(const Feature &rhs) const
    {
        if (this->H() != rhs.H() || this->W() != rhs.W() || this->C() != rhs.C())
            return false;
        for (size_t i = 0; i < this->H(); ++i)
        {
            for (size_t j = 0; j < this->W(); ++j)
            {
                for (size_t k = 0; k < this->C(); ++k)
                {
                    if (this->at(i, j, k) != rhs.at(i, j, k))
                        return false;
                }
            }
        }
        return true;
    }
    // original part is placed up-left, while down-right are still ret's initialized 0s
    Feature pad2Multiple(size_t H0, size_t W0, size_t C0) const
    {
        size_t rh = (this->H() + H0 - 1) / H0 * H0;
        size_t rw = (this->W() + W0 - 1) / W0 * W0;
        size_t rc = (this->C() + C0 - 1) / C0 * C0;
        Feature ret(rh, rw, rc);
        for (size_t i = 0; i < H(); ++i)
        {
            for (size_t j = 0; j < W(); ++j)
            {
                for (size_t k = 0; k < C(); ++k)
                {
                    ret.at(i, j, k) = at(i, j, k);
                }
            }
        }
        return ret;
    };
    // this padding function pad the feature matrix surrounded with 0s
    Feature subPad(size_t h1, size_t w1, size_t c1, size_t H0, size_t W0, size_t C0, size_t row_pad, size_t col_pad) const
    {
        assert(this->H() % H0 == 0);
        assert(this->W() % W0 == 0);
        assert(this->C() % C0 == 0);
        assert((h1 + 1) * H0 <= this->H()); // H1*H0 = H(); H0 is a global constant; h1 is an outside iterator
        assert((w1 + 1) * W0 <= this->W()); // W1*W0 = W(); W0 is a global constant; w1 is an outside iterator
        assert((c1 + 1) * C0 <= this->C()); // C1*C0 = C(); C0 is a global constant; c1 is an outside iterator
        auto H_MAX = H() / H0 - 1;
        auto W_MAX = W() / W0 - 1;
        auto ret = Feature(H0 + row_pad * 2, W0 + col_pad * 2, C0);
#define MOVEorZERO(cond, i, j, k) ret.at((i), (j), (k)) = (cond) ? 0 : at(h1 * H0 + (i) - row_pad, w1 * W0 + (j) - col_pad, c1 * C0 + k)
        for (size_t k = 0; k < C0; ++k)
        {
            for (size_t i = 0; i < row_pad; ++i)
            {
                for (size_t j = 0; j < col_pad; ++j)
                {
                    // x o o
                    // o o o
                    // o o o
                    MOVEorZERO(h1 == 0 || w1 == 0, i, j, k);
                    // o o x
                    // o o o
                    // o o o
                    MOVEorZERO(h1 == 0 || w1 == W_MAX, i, col_pad + W0 + j, k);
                    // o o o
                    // o o o
                    // x o o
                    MOVEorZERO(h1 == H_MAX || w1 == 0, row_pad + H0 + i, j, k);
                    // o o o
                    // o o o
                    // o o x
                    MOVEorZERO(h1 == H_MAX || w1 == W_MAX, row_pad + H0 + i, col_pad + W0 + j, k);
                }
            }
        }

        for (size_t k = 0; k < C0; ++k)
        {
            for (size_t i = 0; i < row_pad; ++i)
            {
                for (size_t j = 0; j < W0; ++j)
                {
                    // o x o
                    // o o o
                    // o o o
                    MOVEorZERO(h1 == 0, i, col_pad + j, k);
                    // o o o
                    // o o o
                    // o x o
                    MOVEorZERO(h1 == H_MAX, row_pad + H0 + i, col_pad + j, k);
                }
            }
        }
        for (size_t k = 0; k < C0; ++k)
        {
            for (size_t i = 0; i < H0; ++i)
            {
                for (size_t j = 0; j < col_pad; ++j)
                {
                    // o o o
                    // x o o
                    // o o o
                    MOVEorZERO(w1 == 0, row_pad + i, j, k);
                    // o o o
                    // o o x
                    // o o o
                    MOVEorZERO(w1 == W_MAX, row_pad + i, col_pad + W0 + j, k);
                }
            }
        }

        for (size_t k = 0; k < C0; ++k)
        {
            for (size_t i = 0; i < H0; ++i)
            {
                for (size_t j = 0; j < W0; ++j)
                {
                    // o o o
                    // o x o
                    // o o o
                    MOVEorZERO(false, row_pad + i, col_pad + j, k);
                }
            }
        }
        return ret;
    }
    // subFeatureAdd adds a H0*W0*C0 subfeature to a H*W*C feature begin at (h1*H0, w1*W0, c1*C0)
    void subAdd(size_t h1, size_t w1, size_t c1, const Feature &sub)
    {
        auto H0 = sub.H(), W0 = sub.W(), C0 = sub.C(); // sub is a H0*W0*C0 feature
        assert(this->H() % H0 == 0);
        assert(this->W() % W0 == 0);
        assert(this->C() % C0 == 0);
        assert((h1 + 1) * H0 <= this->H()); // H1*H0 = H(); H0 is a global constant; h1 is an outside iterator
        assert((w1 + 1) * W0 <= this->W()); // W1*W0 = W(); W0 is a global constant; w1 is an outside iterator
        assert((c1 + 1) * C0 <= this->C()); // C1*C0 = C(); C0 is a global constant; c1 is an outside iterator
        for (size_t i = 0; i < H0; ++i)
        {
            for (size_t j = 0; j < W0; ++j)
            {
                for (size_t k = 0; k < C0; ++k)
                {
                    this->at(h1 * H0 + i, w1 * W0 + j, c1 * C0 + k) += sub.at(i, j, k);
                }
            }
        }
    }
    // transform the padded sub feature into a left hand Mat
    Mat transform(size_t Kh, size_t Kw, size_t stride) const
    {
        assert(Kh % 2 == 1); // kernel should be a square matrix whose sidelength should be an odd number
        assert(Kw % 2 == 1);
        size_t retH = (this->H() - Kh) / stride + 1;
        size_t retW = (this->W() - Kw) / stride + 1;
        Mat ret(retH * retW, this->C() * Kh * Kw);
        for (size_t i = 0; i < retH; ++i)
        {
            for (size_t j = 0; j < retW; ++j)
            {
                for (size_t c = 0; c < this->C(); ++c)
                {
                    for (size_t ki = 0; ki < Kh; ++ki)
                    {
                        for (size_t kj = 0; kj < Kw; ++kj)
                        {
                            ret.at(i * retW + j, c * Kh * Kw + ki * Kw + kj) = this->at(i * stride + ki, j * stride + kj, c);
                        }
                    }
                }
            }
        }
        return ret;
    }
    // select *this feature from (0,0,0) to (selH, selW, selC)
    Feature range(size_t newH, size_t newW, size_t newC) const
    {
        assert(newH <= H());
        assert(newW <= W());
        assert(newC <= C());
        auto ret = Feature(newH, newW, newC);
        for (size_t i = 0; i < newH; ++i)
        {
            for (size_t j = 0; j < newW; ++j)
            {
                for (size_t k = 0; k < newC; ++k)
                {
                    ret.at(i, j, k) = at(i, j, k);
                }
            }
        }
        return ret;
    }
};

// invTrans transform a Mat(whose each column stands for an output channel) into a Feature: W->Co, H->H0*W0
Feature invTrans(size_t Ho, size_t Wo, size_t Co, Mat &orig)
{
    Feature ret(Ho, Wo, Co);
    for (size_t k = 0; k < Co; ++k)
    {
        for (size_t i = 0; i < Ho; ++i)
        {
            for (size_t j = 0; j < Wo; ++j)
            {
                ret.at(i, j, k) = orig.at(i * Wo + j, k);
            }
        }
    }
    return ret;
}

// golden_FWconv
Feature golden_WconvF(Feature f, Weight w, size_t stride)
{
    assert(w.Kh() % 2 == 1); // kernel should be a square matrix whose sidelength should be an odd number
    assert(w.Kw() % 2 == 1);
    size_t retH = (f.H() - w.Kh() + w.Kh() / 2 * 2) / stride + 1;
    size_t retW = (f.W() - w.Kw() + w.Kw() / 2 * 2) / stride + 1;
    Feature res(retH, retW, w.Co());
    auto row_pad = w.Kh() / 2;
    auto col_pad = w.Kw() / 2;
    for (size_t co = 0; co < w.Co(); ++co)
    {
        for (size_t i = 0; i < retH; ++i)
        {
            for (size_t j = 0; j < retW; ++j)
            {
                for (size_t ci = 0; ci < w.Ci(); ++ci)
                {
                    for (size_t ki = 0; ki < w.Kh(); ++ki)
                    {
                        for (size_t kj = 0; kj < w.Kw(); ++kj)
                        {
                            res.at(i, j, co) += f.atElseZero(stride * i - row_pad + ki, stride * j - col_pad + kj, ci) * w.at(co, ki, kj, ci);
                        }
                    }
                }
            }
        }
    }
    return res;
};

#endif // FEATURE_HPP