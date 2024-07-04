#include "CU.hpp"
#include "DLU.hpp"
#include "Feature.hpp"
#include "LDTU.hpp"
#include "RDTU.hpp"
#include "Weight.hpp"
#include <iostream>

L2 l2_i(1024 * 1024);              // unit: 1 byte
DLU dlu_i(l2_i, 64 * 1024);        // unit: 16 bytes
LDTU ldtu_i(dlu_i, 4 * 1024);      // unit: 16 bytes
RDTU rdtu_i(dlu_i, 4 * 1024);      // unit: 16 bytes
CU cu_i(ldtu_i, rdtu_i, 8 * 1024); // unit: 16 bytes

#define CEIL(X, M) ((X + M - 1) / M)

#define FEATURE_INDEX(H, W, C, i, j, k) (i * W * C + j * C + k)
#define WEIGHT_INDEX(Co, Kh, Kw, Ci, oi, ki, kj, ii) \
    (oi * Kh * Kw * Ci + ki * Kw * Ci + kj * Ci + ii)

// #ifdef ROADMAP
int main()
{
    // ----- Feature in L2 -----
    size_t feature_l2_addr = 0;
    size_t H = 20;
    size_t W = 21;
    size_t C = 17;

    Feature feature(H, W, C);
    for (size_t i = 0; i < H; ++i)
        for (size_t j = 0; j < W; ++j)
            for (size_t k = 0; k < C; ++k)
            {
                auto val = i + j * H + k * W * H;
                l2_i.write(feature_l2_addr + FEATURE_INDEX(H, W, C, i, j, k), val);
                feature.at(i, j, k) = val;
            }

    // ----- Weight in L2 -----
    size_t weight_l2_addr = H * W * C;
    size_t Co = 17;
    size_t Kh = 5;
    size_t Kw = Kh;
    size_t Ci = C;

    Weight weight(Co, Kh, Kw, Ci);
    for (size_t oi = 0; oi < Co; ++oi)
        for (size_t ki = 0; ki < Kh; ++ki)
            for (size_t kj = 0; kj < Kw; ++kj)
                for (size_t ii = 0; ii < Ci; ++ii)
                {
                    auto val = ki + kj * oi + ii;
                    l2_i.write(weight_l2_addr + WEIGHT_INDEX(Co, Kh, Kw, Ci, oi, ki, kj, ii),
                               val);
                    weight.at(oi, ki, kj, ii) = val;
                }

    // The task to be done here should be the one scheduled by the SU
    // ---- configure subFeature/subWeight in L1 -----
    size_t H0 = 8;
    size_t W0 = 3;
    size_t Co0 = 5;

    size_t feature_l1_addr = 0;
    dlu_i.configureFeature(feature_l2_addr, H, W, C, H0, W0);

    size_t weight_l1_addr = H0 * W0 * Kh * Kw;
    dlu_i.configureWeight(weight_l2_addr, Co, Kh, Kw, Ci, Co0);

    // ----- configure feature in LMB -----
    ldtu_i.configureFeature(feature_l1_addr, H0, W0, Kh, Kw);

    // ----- configure weight in RMB -----
    rdtu_i.configureWeight(weight_l1_addr, Co0, Kh, Kw);

    // ----- configure matrix in CU-----
    cu_i.configureMat(H0 * W0, Kh * Kw, Co0);

    // ----- computation -----
    size_t Co1 = CEIL(Co, Co0);
    size_t H1 = CEIL(H, H0);
    size_t W1 = CEIL(W, W0);
    size_t Ci1 = CEIL(Ci, C0);

    Feature res(H, W, Co); // For easy comparison, use Feature to store PSB transferred results

    for (size_t oi = 0; oi < Co1; ++oi)
        for (size_t i = 0; i < H1; ++i)
            for (size_t j = 0; j < W1; ++j)
            {
                // clear PSB,ii loop is finished and PSB is the result of a (H0,W0,Co0) sub-block
                // if Co0 < C0 (N < N0) then there will be padding
                cu_i.clearPSB();
                // Continue to do Ci1 times to complete the accumulation of Ci dimensions
                for (size_t ii = 0; ii < Ci1; ++ii)
                {
                    dlu_i.loadSubPaddedFeature(feature_l1_addr, i, j, ii);
                    dlu_i.loadSubWeight(weight_l1_addr, oi, ii);
                    ldtu_i.load_im2col();
                    rdtu_i.load();

                    // CU calculation
                    // Done computation along the H0*W0 dimension
                    for (size_t i = 0; i < CEIL(H0 * W0, 16); ++i)
                        // finished adding Kh*Kw dimensions
                        for (size_t j = 0; j < Kh * Kw; ++j)
                            // Done computation along the Co0 dimension
                            for (size_t k = 0; k < CEIL(Co0, 16); ++k)
                                cu_i.matmul(i, j, k);
                }

                // PSB transfer (H0,W0,Co0) to Feature
                for (size_t sub_i = 0; sub_i < H0; ++sub_i)
                    for (size_t sub_j = 0; sub_j < W0; ++sub_j)
                        for (size_t co_i = 0; co_i < CEIL(Co0, 16); ++co_i)
                            for (size_t sub_k = 0; sub_k < 16; ++sub_k)
                            {
                                auto im_row = H0 * i + sub_i;
                                auto im_col = W0 * j + sub_j;
                                auto im_ch = (Co0 * oi) + co_i * 16 + sub_k;
                                if (im_row < H && im_col < W && im_ch < Co)
                                    // See CU::matmul for details on how (H0,W0,Co0) is stored in PSB
                                    res.at(im_row, im_col, im_ch) =
                                        cu_i.getPSB()[(sub_i * W0 + sub_j) * CEIL(Co0, 16) + co_i][sub_k];
                            }
            }
    // res.display(8);

    auto ref_result = golden_WconvF(feature, weight, 1);
    // ref_result.display(8);
    assert(res == ref_result);
    puts("Passed");
}
// #endif