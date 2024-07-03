#ifndef WEIGHT_HPP
#define WEIGHT_HPP 1

#include "Mat.hpp"
#include <cassert>
#include <cstring>
#include <format>
#include <iostream>
#include <vector>
#include <algorithm>

class Weight
{
private:
    size_t w_Co;
    size_t w_Kh;
    size_t w_Kw;
    size_t w_Ci;
    std::vector<int8_t> w_data;

public:
    // ctors & dtor
    Weight(size_t Co, size_t Kh, size_t Kw, size_t Ci) // constructor use Co, Kh, Kw, Ci and fill the w_data with 0s
        : w_Co{Co}, w_Kh{Kh}, w_Kw{Kw}, w_Ci{Ci}, w_data(Co * Kh * Kw * Ci)
    {
        std::fill(std::begin(w_data), std::end(w_data), 0);
    }
    Weight(const Weight &other) // copy constructor
        : w_Co{other.Co()}, w_Kh{other.Kh()}, w_Kw{other.Kw()}, w_Ci{other.Ci()}, w_data{other.w_data}
    {
    }
    /*Weight(Weight &&other) noexcept // move constructor
        : w_Co{other.Co()}, w_Kh{other.Kh()}, w_Kw{other.Kw()}, w_Ci{other.Ci()}, w_data{other.w_data}
    {
        other.w_Co = 0;
        other.w_Kh = 0;
        other.w_Kw = 0;
        other.w_Ci = 0;
        std::vector<int8_t> empty;
        other.w_data = empty;
    }*/
    Weight &operator=(const Weight &other) // copy assignment operator
    {
        w_Co = other.Co();
        w_Kh = other.Kh();
        w_Kw = other.Kw();
        w_Ci = other.Ci();
        w_data = other.w_data;
        return *this;
    }
    /*Weight &operator=(Weight &&other) noexcept // move assignment operator
    {
        if (this != &other)
        {
            w_Co = other.Co();
            w_Kh = other.Kh();
            w_Kw = other.Kw();
            w_Ci = other.Ci();
            w_data = other.w_data;
            other.w_Co = 0;
            other.w_Kh = 0;
            other.w_Kw = 0;
            other.w_Ci = 0;
            std::vector<int8_t> empty;
            other.w_data = empty;
        }
        return *this;
    }*/
    ~Weight() {} // destructor
    // getters
    size_t Co() const
    {
        return w_Co;
    }
    size_t Kh() const
    {
        return w_Kh;
    }
    size_t Kw() const
    {
        return w_Kw;
    }
    size_t Ci() const
    {
        return w_Ci;
    }
    int8_t at(size_t co, size_t kh, size_t kw, size_t ci) const // an overloading function of four-dimensional .at() which can only read the value
    {
        assert(co < Co());
        assert(kh < Kh());
        assert(kw < Kw());
        assert(ci < Ci());
        return w_data[co * Kh() * Kw() * Ci() + kh * Kw() * Ci() + kw * Ci() + ci];
    }
    // setter
    int8_t &at(size_t co, size_t kh, size_t kw, size_t ci) // an overloading function of four-dimensional .at() which can only read the value
    {
        assert(co < Co());
        assert(kh < Kh());
        assert(kw < Kw());
        assert(ci < Ci());
        return w_data[co * Kh() * Kw() * Ci() + kh * Kw() * Ci() + kw * Ci() + ci];
    }
    // transform Weight to Mat, each col is corresponding to a flatted set of Ci weights, namely, a set for a Co channel
    Mat transform() const
    {
        Mat ret(this->Ci() * this->Kh() * this->Kw(), this->Co());
        for (size_t co = 0; co < this->Co(); ++co)
        {
            for (size_t kh = 0; kh < this->Kh(); ++kh)
            {
                for (size_t kw = 0; kw < this->Kw(); ++kw)
                {
                    for (size_t ci = 0; ci < this->Ci(); ++ci)
                    {
                        ret.at(ci * Kh() * Kw() + kh * Kw() + kw, co) = this->at(co, kh, kw, ci);
                    }
                }
            }
        }
        return ret;
    }
    // pad to multiple, original part is placed up-left, while down-right are still ret's initialized 0s
    Weight pad2Multiple(size_t Co0, size_t Ci0) const
    {
        size_t rco = (this->Co() + Co0 - 1) / Co0 * Co0;
        size_t rci = (this->Ci() + Ci0 - 1) / Ci0 * Ci0;
        Weight ret(rco, this->Kh(), this->Kw(), rci);
        for (size_t co = 0; co < this->Co(); ++co)
        {
            for (size_t kh = 0; kh < this->Kh(); ++kh)
            {
                for (size_t kw = 0; kw < this->Kw(); ++kw)
                {
                    for (size_t ci = 0; ci < this->Ci(); ++ci)
                    {
                        ret.at(co, kh, kw, ci) = at(co, kh, kw, ci);
                    }
                }
            }
        }
        return ret;
    }
    // subweight returns a Co0*Kh*Kw*Ci0 matrix resulted from divided Co*Kh*Kw*Ci Weight
    Weight subWeight(size_t co1, size_t ci1, size_t Co0, size_t Ci0) const
    {
        Weight ret(Co0, this->Kh(), this->Kw(), Ci0); // Co0 and Ci0 are global constants; co1 and ci1 are outside iterators;
        for (size_t co = 0; co < Co0; ++co)
        {
            for (size_t kh = 0; kh < this->Kh(); ++kh)
            {
                for (size_t kw = 0; kw < this->Kw(); ++kw)
                {
                    for (size_t ci = 0; ci < Ci0; ++ci)
                    {
                        ret.at(co, kh, kw, ci) = at(co1 * Co0 + co, kh, kw, ci1 * Ci0 + ci);
                    }
                }
            }
        }
        return ret;
    }
};

#endif // WEIGHT_HPP