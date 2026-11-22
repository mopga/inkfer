// Far-range attention of a decode / verification call on the tensor cores (prototype for kernels.cu).
//
// The far keys [0, far) of a call are the keys every row sees at a distance past the relative-bias
// table (far = min over rows of position - extent + 1, rounded down to whole 16-key tiles): no
// bias, no mask, the same keys for every (row, head) pair of a KV head. One block streams a
// contiguous share of them for one KV head and up to eight rows (24 pairs), so each key's codes
// cross the bus once per call instead of once per row.
//
// Per warp and 16-key tile, with the scores transposed (keys are the mma's M, pairs its N):
//   S^T[16 keys][8 pairs] = K[16][128] Q^T          (K halves as the scalar kernel decodes them,
//                                                     Q in FP32 as hi + lo FP16 halves)
//   O^T[128 dims][8 pairs] += V^T[128][16] P^T      (P in FP32 as hi + lo FP16 halves)
// accumulated in FP32. A dimension permutation lets every thread read whole bytes of its rows:
// QK step j gives thread c dimensions 32c + 4j .. +3, PV tile mt gives thread g dimensions
// 16g + 2mt, +1. A warp keeps its own online softmax; the block merges its four warps and writes
// one partial (m, l, acc) per pair into the decode partial slots the combine kernel merges.

#include <cuda_fp16.h>

namespace ninfer::models::inkling2::kernels {

constexpr int kFarTile   = 16; // keys per warp step
constexpr int kFarWarps  = 4;
constexpr int kFarStages = 3;
constexpr int kFarRows   = 8;  // rows per row group: 24 pairs, three n-tiles

__device__ __forceinline__ void far_cp16(void* smem, const void* global) {
    const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(global));
}
__device__ __forceinline__ void far_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void far_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ unsigned far_trans(unsigned x) {
    unsigned y;
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
    return y;
}

__device__ __forceinline__ void far_mma(float (&d)[4], unsigned a0, unsigned a1, unsigned a2,
                                        unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned*>(&h); }
__device__ __forceinline__ __half2 u2h(unsigned u) { return *reinterpret_cast<__half2*>(&u); }

// Signed bytes b0, b1 of `bytes` (selected by prmt selector `sel`, byte 0 and byte 2 of the
// result) as a half2: 0x6400 | (b ^ 0x80) is the half 1152 + b, exact.
__device__ __forceinline__ __half2 far_i8x2(unsigned flipped, unsigned sel) {
    const unsigned h = __byte_perm(flipped, 0x64646464u, sel);
    return __hsub2(u2h(h), __half2half2(__ushort_as_half(0x6480))); // 1152
}

// The physical end of the far range: keys [0, far) of the call, whole tiles.
__device__ __forceinline__ int far_keys(const AttentionArgs& a) {
    int f = INT_MAX;
    for (int t = 0; t < a.rows.tokens; ++t) { f = min(f, a.rows.positions[t] - a.extent + 1); }
    if (a.rows.chunk) { f = min(f, a.rows.first); }
    return max(0, f) / kFarTile * kFarTile;
}

template <KvCoding KC>
struct FarLayout {
    static constexpr int kKCode  = KC == KvCoding::Lloyd4 ? 64 : 128;
    static constexpr int kKScale = 4, kVCode = 64, kVScale = 8;
    static constexpr int kKCodeAt = 0;
    static constexpr int kKScaleAt = kKCodeAt + kFarTile * kKCode;
    static constexpr int kVCodeAt = kKScaleAt + kFarTile * kKScale;
    static constexpr int kVScaleAt = kVCodeAt + kFarTile * kVCode;
    static constexpr int kStage = kVScaleAt + kFarTile * kVScale;
    static constexpr int kChunks = kStage / 16;
};

template <int NT>
constexpr int far_smem_bytes(int stage) {
    const int stages = kFarWarps * kFarStages * stage;
    const int q      = NT * 8 * 32 * 16;
    const int merge  = NT * 8 * 128 * 4 + 2 * kFarWarps * NT * 8 * 4;
    return q + (stages > merge ? stages : merge);
}

// Block (split, kv head, row group). Partial slot slot0 + split of every pair of the group.
template <KvCoding KC, int NT, bool kSplitP>
__global__ void __launch_bounds__(128) far_attention_kernel(AttentionArgs a, int far_splits,
                                                            int slot0, int total_splits) {
    using L = FarLayout<KC>;
    extern __shared__ __align__(16) std::uint8_t smem[];
    uint4* q_s           = reinterpret_cast<uint4*>(smem);         // [NT][8][32]
    std::uint8_t* stages = smem + NT * 8 * 32 * 16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, c = lane & 3;
    const int kv_head = blockIdx.y;
    const int group   = a.query_heads / a.kv_heads;
    const int r0      = blockIdx.z * kFarRows;
    const int rows    = min(kFarRows, a.rows.tokens - r0);
    const int pairs   = rows * group;

    __shared__ int far_s;
    if (threadIdx.x == 0) { far_s = far_keys(a); }
    // Q fragments, hi and lo: entry (nt, j, lane) = pair 8 nt + g, dimensions 32 c + 4 j .. + 3.
    for (int e = threadIdx.x; e < NT * 8 * 32; e += blockDim.x) {
        const int nt = e / 256, j = (e / 32) % 8, el = e % 32;
        const int pair = 8 * nt + (el >> 2), ec = el & 3;
        uint4 w = make_uint4(0, 0, 0, 0);
        if (pair < pairs) {
            const int row  = r0 + pair / group;
            const int head = kv_head * group + pair % group;
            float4 v = reinterpret_cast<const float4*>(
                a.q + (std::int64_t(row) * a.query_heads + head) * kD + 32 * ec)[j];
            v.x *= a.scale; v.y *= a.scale; v.z *= a.scale; v.w *= a.scale;
            const __half2 h0 = __floats2half2_rn(v.x, v.y), h1 = __floats2half2_rn(v.z, v.w);
            const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
            const __half2 l0 = __floats2half2_rn(v.x - f0.x, v.y - f0.y);
            const __half2 l1 = __floats2half2_rn(v.z - f1.x, v.w - f1.y);
            w = make_uint4(h2u(h0), h2u(h1), h2u(l0), h2u(l1));
        }
        q_s[e] = w;
    }
    __syncthreads();
    const int tiles = far_s / kFarTile;
    const int per   = (tiles + far_splits - 1) / far_splits;
    const int tb    = blockIdx.x * per, te = min(tiles, tb + per);
    const int sequence = a.rows.sequence;
    const KvPlaneView& kp = a.k_plane;
    const KvPlaneView& vp = a.v_plane;
    const std::uint8_t* k_codes = kp.codes + std::int64_t(sequence) * kp.code_seq + std::int64_t(kv_head) * kp.code_head;
    const std::uint8_t* k_scales = kp.scales + std::int64_t(sequence) * kp.scale_seq + std::int64_t(kv_head) * kp.scale_head;
    const std::uint8_t* v_codes = vp.codes + std::int64_t(sequence) * vp.code_seq + std::int64_t(kv_head) * vp.code_head;
    const std::uint8_t* v_scales = vp.scales + std::int64_t(sequence) * vp.scale_seq + std::int64_t(kv_head) * vp.scale_head;
    std::uint8_t* mine = stages + warp * kFarStages * L::kStage;

    auto load = [&](int tile, int buffer) {
        std::uint8_t* st = mine + buffer * L::kStage;
        const std::int64_t key = std::int64_t(tile) * kFarTile;
        for (int ch = lane; ch < L::kChunks; ch += 32) {
            const int at = ch * 16;
            const std::uint8_t* src;
            if (at < L::kKScaleAt) { src = k_codes + key * L::kKCode + at; }
            else if (at < L::kVCodeAt) { src = k_scales + key * L::kKScale + (at - L::kKScaleAt); }
            else if (at < L::kVScaleAt) { src = v_codes + key * L::kVCode + (at - L::kVCodeAt); }
            else { src = v_scales + key * L::kVScale + (at - L::kVScaleAt); }
            far_cp16(st + at, src);
        }
    };

    float o[8][NT][4], m[NT][2], l[NT][2];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        m[nt][0] = m[nt][1] = -INFINITY;
        l[nt][0] = l[nt][1] = 0.0f;
#pragma unroll
        for (int mt = 0; mt < 8; ++mt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { o[mt][nt][i] = 0.0f; }
        }
    }
    const int first = tb + warp;
    const int count = first < te ? (te - first + kFarWarps - 1) / kFarWarps : 0;
#pragma unroll
    for (int s = 0; s < kFarStages - 1; ++s) {
        if (s < count) { load(first + s * kFarWarps, s); }
        far_commit();
    }
    for (int k = 0; k < count; ++k) {
        const int ahead = k + kFarStages - 1;
        if (ahead < count) { load(first + ahead * kFarWarps, ahead % kFarStages); }
        far_commit();
        far_wait<kFarStages - 1>();
        __syncwarp();
        const std::uint8_t* st = mine + (k % kFarStages) * L::kStage;

        // ---- S^T = K Q^T
        float s[NT][4];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f; }
        const __half* ks = reinterpret_cast<const __half*>(st + L::kKScaleAt);
        // Scale group of dimensions 32 c .. 32 c + 31: c / 2 (both codings' groups are 64 wide).
        const __half2 sg  = __half2half2(ks[g * 2 + (c >> 1)]);
        const __half2 sg8 = __half2half2(ks[(g + 8) * 2 + (c >> 1)]);
        if constexpr (KC == KvCoding::Lloyd4) {
            const uint4 kg  = *reinterpret_cast<const uint4*>(st + L::kKCodeAt + g * 64 + 16 * c);
            const uint4 kg8 = *reinterpret_cast<const uint4*>(st + L::kKCodeAt + (g + 8) * 64 + 16 * c);
            const unsigned wg[4] = {kg.x, kg.y, kg.z, kg.w}, wg8[4] = {kg8.x, kg8.y, kg8.z, kg8.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned e  = ops::kv_cache_lloyd4_expand4(wg[j >> 1] >> (16 * (j & 1))) ^ 0x80808080u;
                const unsigned e8 = ops::kv_cache_lloyd4_expand4(wg8[j >> 1] >> (16 * (j & 1))) ^ 0x80808080u;
                const unsigned a0 = h2u(__hmul2(far_i8x2(e, 0x7170), sg));
                const unsigned a2 = h2u(__hmul2(far_i8x2(e, 0x7372), sg));
                const unsigned a1 = h2u(__hmul2(far_i8x2(e8, 0x7170), sg8));
                const unsigned a3 = h2u(__hmul2(far_i8x2(e8, 0x7372), sg8));
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const uint4 q = q_s[(nt * 8 + j) * 32 + lane];
                    far_mma(s[nt], a0, a1, a2, a3, q.x, q.y);
                    far_mma(s[nt], a0, a1, a2, a3, q.z, q.w);
                }
            }
        } else {
            const uint4* kg  = reinterpret_cast<const uint4*>(st + L::kKCodeAt + g * 128 + 32 * c);
            const uint4* kg8 = reinterpret_cast<const uint4*>(st + L::kKCodeAt + (g + 8) * 128 + 32 * c);
            const uint4 x0 = kg[0], x1 = kg[1], y0 = kg8[0], y1 = kg8[1];
            const unsigned wg[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            const unsigned wg8[8] = {y0.x, y0.y, y0.z, y0.w, y1.x, y1.y, y1.z, y1.w};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const unsigned e = wg[j] ^ 0x80808080u, e8 = wg8[j] ^ 0x80808080u;
                const unsigned a0 = h2u(__hmul2(far_i8x2(e, 0x7170), sg));
                const unsigned a2 = h2u(__hmul2(far_i8x2(e, 0x7372), sg));
                const unsigned a1 = h2u(__hmul2(far_i8x2(e8, 0x7170), sg8));
                const unsigned a3 = h2u(__hmul2(far_i8x2(e8, 0x7372), sg8));
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const uint4 q = q_s[(nt * 8 + j) * 32 + lane];
                    far_mma(s[nt], a0, a1, a2, a3, q.x, q.y);
                    far_mma(s[nt], a0, a1, a2, a3, q.z, q.w);
                }
            }
        }

        // ---- online softmax per pair (columns 2c, 2c + 1 of each n-tile), P^T fragments
        unsigned pb[NT][2][2]; // [nt][hi, lo][b0, b1]
        float rescale[NT][2];
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            float mx0 = fmaxf(s[nt][0], s[nt][2]), mx1 = fmaxf(s[nt][1], s[nt][3]);
#pragma unroll
            for (int off = 4; off < 32; off <<= 1) {
                mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
                mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
            }
            const float n0 = fmaxf(m[nt][0], mx0), n1 = fmaxf(m[nt][1], mx1);
            rescale[nt][0] = expf(m[nt][0] - n0);
            rescale[nt][1] = expf(m[nt][1] - n1);
            m[nt][0] = n0;
            m[nt][1] = n1;
            const float p0 = expf(s[nt][0] - n0), p1 = expf(s[nt][1] - n1);
            const float p2 = expf(s[nt][2] - n0), p3 = expf(s[nt][3] - n1);
            l[nt][0] = l[nt][0] * rescale[nt][0] + (p0 + p2);
            l[nt][1] = l[nt][1] * rescale[nt][1] + (p1 + p3);
            const __half2 h01 = __floats2half2_rn(p0, p1), h23 = __floats2half2_rn(p2, p3);
            pb[nt][0][0] = far_trans(h2u(h01));
            pb[nt][0][1] = far_trans(h2u(h23));
            if constexpr (kSplitP) {
                const float2 f01 = __half22float2(h01), f23 = __half22float2(h23);
                pb[nt][1][0] = far_trans(h2u(__floats2half2_rn(p0 - f01.x, p1 - f01.y)));
                pb[nt][1][1] = far_trans(h2u(__floats2half2_rn(p2 - f23.x, p3 - f23.y)));
            }
        }

        // ---- O^T += V^T P^T: thread g's dimensions 16 g .. 16 g + 15 of keys 2c, 2c+1, 2c+8, 2c+9
        const std::uint8_t* vc = st + L::kVCodeAt;
        const __half* vs       = reinterpret_cast<const __half*>(st + L::kVScaleAt);
        const uint2 va = *reinterpret_cast<const uint2*>(vc + (2 * c) * 64 + 8 * g);
        const uint2 vb = *reinterpret_cast<const uint2*>(vc + (2 * c + 1) * 64 + 8 * g);
        const uint2 vc8 = *reinterpret_cast<const uint2*>(vc + (2 * c + 8) * 64 + 8 * g);
        const uint2 vd = *reinterpret_cast<const uint2*>(vc + (2 * c + 9) * 64 + 8 * g);
        const __half2 sab = __halves2half2(vs[(2 * c) * 4 + (g >> 1)], vs[(2 * c + 1) * 4 + (g >> 1)]);
        const __half2 scd = __halves2half2(vs[(2 * c + 8) * 4 + (g >> 1)], vs[(2 * c + 9) * 4 + (g >> 1)]);
        const unsigned A[2] = {va.x ^ 0x88888888u, va.y ^ 0x88888888u};
        const unsigned B[2] = {vb.x ^ 0x88888888u, vb.y ^ 0x88888888u};
        const unsigned C[2] = {vc8.x ^ 0x88888888u, vc8.y ^ 0x88888888u};
        const unsigned D[2] = {vd.x ^ 0x88888888u, vd.y ^ 0x88888888u};
        const __half2 bias = __half2half2(__ushort_as_half(0x6408)); // 1032
#pragma unroll
        for (int mt = 0; mt < 8; ++mt) {
            // byte mt of each key's eight: dimensions 16 g + 2 mt (low nibble) and + 1 (high).
            const unsigned b   = mt & 3;
            const unsigned sel = b | (b << 4) | ((4 + b) << 8) | ((4 + b) << 12);
            const unsigned abw = __byte_perm(A[mt >> 2], B[mt >> 2], sel); // A_b, A_b, B_b, B_b
            const unsigned cdw = __byte_perm(C[mt >> 2], D[mt >> 2], sel);
            const unsigned a0 = h2u(__hmul2(__hsub2(u2h((abw & 0x000f000fu) | 0x64006400u), bias), sab));
            const unsigned a1 = h2u(__hmul2(__hsub2(u2h(((abw >> 4) & 0x000f000fu) | 0x64006400u), bias), sab));
            const unsigned a2 = h2u(__hmul2(__hsub2(u2h((cdw & 0x000f000fu) | 0x64006400u), bias), scd));
            const unsigned a3 = h2u(__hmul2(__hsub2(u2h(((cdw >> 4) & 0x000f000fu) | 0x64006400u), bias), scd));
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                o[mt][nt][0] *= rescale[nt][0];
                o[mt][nt][1] *= rescale[nt][1];
                o[mt][nt][2] *= rescale[nt][0];
                o[mt][nt][3] *= rescale[nt][1];
                far_mma(o[mt][nt], a0, a1, a2, a3, pb[nt][0][0], pb[nt][0][1]);
                if constexpr (kSplitP) { far_mma(o[mt][nt], a0, a1, a2, a3, pb[nt][1][0], pb[nt][1][1]); }
            }
        }
        __syncwarp();
    }
    far_wait<0>();

    // ---- merge the four warps: l over the warp's key rows, then the block's warps in order
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int off = 4; off < 32; off <<= 1) {
            l[nt][0] += __shfl_xor_sync(0xffffffffu, l[nt][0], off);
            l[nt][1] += __shfl_xor_sync(0xffffffffu, l[nt][1], off);
        }
    }
    __syncthreads(); // stages are free
    float* o_s = reinterpret_cast<float*>(stages);          // [NT*8][128]
    float* m_s = o_s + NT * 8 * 128;                          // [warps][NT*8]
    float* l_s = m_s + kFarWarps * NT * 8;
    if (g == 0) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            m_s[warp * NT * 8 + 8 * nt + 2 * c]     = m[nt][0];
            m_s[warp * NT * 8 + 8 * nt + 2 * c + 1] = m[nt][1];
            l_s[warp * NT * 8 + 8 * nt + 2 * c]     = l[nt][0];
            l_s[warp * NT * 8 + 8 * nt + 2 * c + 1] = l[nt][1];
        }
    }
    for (int i = threadIdx.x; i < NT * 8 * 128; i += blockDim.x) { o_s[i] = 0.0f; }
    __syncthreads();
    float wt[NT][2];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int pair = 8 * nt + 2 * c + h;
            float top = -INFINITY;
            for (int w = 0; w < kFarWarps; ++w) { top = fmaxf(top, m_s[w * NT * 8 + pair]); }
            wt[nt][h] = m[nt][h] == -INFINITY ? 0.0f : expf(m[nt][h] - top);
        }
    }
    for (int w = 0; w < kFarWarps; ++w) {
        if (warp == w) {
#pragma unroll
            for (int mt = 0; mt < 8; ++mt) {
#pragma unroll
                for (int nt = 0; nt < NT; ++nt) {
                    const int p0 = 8 * nt + 2 * c, d0 = 16 * g + 2 * mt;
                    o_s[p0 * 128 + d0]           += o[mt][nt][0] * wt[nt][0];
                    o_s[(p0 + 1) * 128 + d0]     += o[mt][nt][1] * wt[nt][1];
                    o_s[p0 * 128 + d0 + 1]       += o[mt][nt][2] * wt[nt][0];
                    o_s[(p0 + 1) * 128 + d0 + 1] += o[mt][nt][3] * wt[nt][1];
                }
            }
        }
        __syncthreads();
    }
    // ---- one partial per pair: slot slot0 + split
    const std::int64_t prow_all = std::int64_t(a.rows.tokens) * a.query_heads * total_splits;
    for (int pair = 0; pair < pairs; ++pair) {
        const int row  = r0 + pair / group;
        const int head = kv_head * group + pair % group;
        const std::int64_t prow = (std::int64_t(row) * a.query_heads + head) * total_splits + slot0 + blockIdx.x;
        float top = -INFINITY;
        for (int w = 0; w < kFarWarps; ++w) { top = fmaxf(top, m_s[w * NT * 8 + pair]); }
        a.partial[prow * kD + threadIdx.x] = o_s[pair * 128 + threadIdx.x];
        if (threadIdx.x == 0) {
            float sum = 0.0f;
            for (int w = 0; w < kFarWarps; ++w) {
                const float mw = m_s[w * NT * 8 + pair];
                if (mw != -INFINITY) { sum += l_s[w * NT * 8 + pair] * expf(mw - top); }
            }
            a.partial[prow_all * kD + 2 * prow]     = top;
            a.partial[prow_all * kD + 2 * prow + 1] = sum;
        }
    }
}

} // namespace ninfer::models::inkling2::kernels
