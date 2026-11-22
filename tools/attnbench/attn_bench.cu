// Inkling 2 full-attention kernel timing and qualification at real shapes (24 query heads, 8 KV
// heads of 128, rk4v4 or rk8v4 planes), outside the engine: the engine's own kernels.cu compiled
// into this program, called through kernels::attention() exactly as executor.cpp calls it.
//
//   attn_bench DEPTH [--kv rk4v4|rk8v4] [--chunk 2048] [--rows 1,2,4,8] [--splits N]
//              [--near N] [--far N] [--cfar N] [--reps 5] [--check 1] [--qscale 0.6]
//
// Prints the time of one prompt chunk's attention (rows DEPTH-chunk .. DEPTH-1 after a cache of
// DEPTH-chunk keys; scalar, and with --cfar tensor-core slots over the cache's far keys, default
// 1, 0 none) and of decode / verification calls of 1..8 rows ending at depth DEPTH, per visit,
// for the scalar route (--splits ranges over every key; default the executor's
// decode_splits_full) and the far-key route (--near ranges over the nearer keys, --far tensor-core
// slots over the rest; defaults the executor's min(splits, 64) + the rest of 128). With --check,
// every output is compared with an FP64 oracle of the whole attention (relative bias, sink, gate,
// the chunk's causal mask) computed from the planes' codes and stored scales, decoded
// independently.
// The model has 8 full-attention visits a token, so a call's attention cost is 8x the printed time.
#include "models/inkling2/kernels.cu"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace ninfer {
void cuda_check(cudaError_t err, const char* expr, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s:%d %s: %s\n", file, line, expr, cudaGetErrorString(err));
        std::exit(2);
    }
}
void raise_dynamic_shared_memory(const void* kernel, int bytes) {
    CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
}
} // namespace ninfer

using namespace ninfer::models::inkling2;
namespace k = ninfer::models::inkling2::kernels;

__global__ void fill_bytes(std::uint8_t* p, std::size_t n, unsigned seed) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        unsigned x = unsigned(i) * 2654435761u ^ seed;
        x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
        p[i] = std::uint8_t(x);
    }
}
__global__ void fill_half(__half* p, std::size_t n, float v) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        p[i] = __float2half(v * (1.0f + 0.25f * float((i * 7) % 5) / 5.0f));
    }
}
__global__ void fill_float(float* p, std::size_t n, float scale, unsigned seed) {
    for (std::size_t i = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x; i < n;
         i += std::size_t(gridDim.x) * blockDim.x) {
        unsigned x = unsigned(i) * 2246822519u ^ seed;
        x ^= x >> 15; x *= 0x2c1b3c6du; x ^= x >> 12;
        p[i] = scale * (float(x & 0xffff) / 32768.0f - 1.0f);
    }
}

static void* dev(std::size_t bytes) {
    void* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, bytes));
    return p;
}

template <class T>
static std::vector<T> host(const void* p, std::size_t n) {
    std::vector<T> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

static int lloyd(int i) {
    static const int codes[8] = {6, 18, 31, 44, 58, 75, 96, 127};
    return (i & 8) ? -codes[i & 7] : codes[i & 7];
}

constexpr int kQH = 24, kKVH = 8, kExtent = 1024, kRel = 16;

// FP64 decode attention of rows at positions pos[0..t) (sequence 0, every key 0..pos in the
// planes): o[t][head][128] = sigmoid(gate) * sum_k e^{s_k} v_k / (sum_k e^{s_k} + e^{sink}),
// s_k = scale q . k + (pos - k < extent ? r . table[pos - k] : 0).
static std::vector<double> oracle(const KvPlaneView& kp, const KvPlaneView& vp, const k::AttentionArgs& a,
                                  const std::vector<int>& pos) {
    const int t = int(pos.size());
    const int D = pos.back() + 1;
    const auto kc = host<std::uint8_t>(kp.codes, std::size_t(kp.code_head) * kKVH);
    const auto ks = host<__half>(kp.scales, std::size_t(kp.scale_head) * kKVH / 2);
    const auto vc = host<std::uint8_t>(vp.codes, std::size_t(vp.code_head) * kKVH);
    const auto vs = host<__half>(vp.scales, std::size_t(vp.scale_head) * kKVH / 2);
    const auto q = host<float>(a.q, std::size_t(t) * kQH * 128);
    const auto r = host<float>(a.r, std::size_t(t) * kQH * kRel);
    const auto sinks = host<float>(a.sinks, std::size_t(t) * kQH);
    const auto table = host<float>(a.table, std::size_t(kExtent) * kRel);
    const auto gate = host<float>(a.gate, std::size_t(t) * a.gate_stride);
    std::vector<double> out(std::size_t(t) * kQH * 128);
    const int group = kQH / kKVH;
    std::vector<double> K(std::size_t(D) * 128), V(std::size_t(D) * 128);
    for (int head = 0; head < kKVH; ++head) {
        for (int key = 0; key < D; ++key) {
            const std::uint8_t* krow = kc.data() + std::size_t(head) * kp.code_head + std::size_t(key) * kp.code_row;
            const __half* kscale = ks.data() + (std::size_t(head) * kp.scale_head + std::size_t(key) * kp.scale_row) / 2;
            const std::uint8_t* vrow = vc.data() + std::size_t(head) * vp.code_head + std::size_t(key) * vp.code_row;
            const __half* vscale = vs.data() + (std::size_t(head) * vp.scale_head + std::size_t(key) * vp.scale_row) / 2;
            for (int d = 0; d < 128; ++d) {
                const int code = kp.coding == KvCoding::Lloyd4 ? lloyd((krow[d / 2] >> (4 * (d & 1))) & 15)
                                                               : int(std::int8_t(krow[d]));
                K[std::size_t(key) * 128 + d] = double(code) * double(__half2float(kscale[d / 64]));
                const int nib = (vrow[d / 2] >> (4 * (d & 1))) & 15;
                V[std::size_t(key) * 128 + d] = double((nib ^ 8) - 8) * double(__half2float(vscale[d / 32]));
            }
        }
        for (int row = 0; row < t; ++row) {
            for (int h = 0; h < group; ++h) {
                const int qh = head * group + h;
                const float* qq = &q[(std::size_t(row) * kQH + qh) * 128];
                const float* rr = &r[(std::size_t(row) * kQH + qh) * kRel];
                std::vector<double> s(pos[row] + 1);
                double m = sinks[std::size_t(row) * kQH + qh];
                for (int key = 0; key <= pos[row]; ++key) {
                    double dot = 0;
                    for (int d = 0; d < 128; ++d) { dot += double(qq[d]) * double(a.scale) * K[std::size_t(key) * 128 + d]; }
                    const int dist = pos[row] - key;
                    if (dist < kExtent) {
                        for (int c = 0; c < kRel; ++c) { dot += double(rr[c]) * double(table[std::size_t(dist) * kRel + c]); }
                    }
                    s[key] = dot;
                    m = std::max(m, dot);
                }
                double l = std::exp(double(sinks[std::size_t(row) * kQH + qh]) - m), acc[128] = {};
                for (int key = 0; key <= pos[row]; ++key) {
                    const double w = std::exp(s[key] - m);
                    l += w;
                    for (int d = 0; d < 128; ++d) { acc[d] += w * V[std::size_t(key) * 128 + d]; }
                }
                for (int d = 0; d < 128; ++d) {
                    const double gx = gate[std::size_t(row) * a.gate_stride + qh * 128 + d];
                    out[(std::size_t(row) * kQH + qh) * 128 + d] = acc[d] / l / (1.0 + std::exp(-gx));
                }
            }
        }
    }
    return out;
}

static void report(const char* name, const __nv_bfloat16* out_dev, const std::vector<double>& ref) {
    const auto got = host<__nv_bfloat16>(out_dev, ref.size());
    double max_abs = 0, max_ref = 0, sq = 0, rsq = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        const double e = std::fabs(double(__bfloat162float(got[i])) - ref[i]);
        max_abs = std::max(max_abs, e);
        max_ref = std::max(max_ref, std::fabs(ref[i]));
        sq += e * e;
        rsq += ref[i] * ref[i];
    }
    std::printf("   %s vs FP64 oracle: max abs %.3e (max |o| %.3e, rel %.3e), rms rel %.3e\n", name, max_abs, max_ref,
                max_abs / max_ref, std::sqrt(sq / rsq));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "attn_bench DEPTH [--kv rk4v4|rk8v4] [--chunk N] [--rows 1,2,4,8] [--splits N] "
                             "[--near N] [--far N] [--reps N] [--check 1] [--qscale X]\n");
        return 1;
    }
    const int depth = std::atoi(argv[1]);
    std::string kv = "rk4v4";
    int chunk = 2048, reps = 5, splits = 0, near_splits = 0, far_splits = 0, check = 0, chunk_far = 1;
    float qscale = 0.6f;
    std::vector<int> rows_list = {1, 2, 4, 6, 8};
    for (int i = 2; i + 1 < argc; i += 2) {
        if (!std::strcmp(argv[i], "--kv")) { kv = argv[i + 1]; }
        else if (!std::strcmp(argv[i], "--chunk")) { chunk = std::atoi(argv[i + 1]); }
        else if (!std::strcmp(argv[i], "--reps")) { reps = std::atoi(argv[i + 1]); }
        else if (!std::strcmp(argv[i], "--splits")) { splits = std::atoi(argv[i + 1]); }
        else if (!std::strcmp(argv[i], "--near")) { near_splits = std::atoi(argv[i + 1]); }
        else if (!std::strcmp(argv[i], "--far")) { far_splits = std::atoi(argv[i + 1]); }
        else if (!std::strcmp(argv[i], "--cfar")) { chunk_far = std::atoi(argv[i + 1]); }
        else if (!std::strcmp(argv[i], "--check")) { check = std::atoi(argv[i + 1]); }
        else if (!std::strcmp(argv[i], "--qscale")) { qscale = float(std::atof(argv[i + 1])); }
        else if (!std::strcmp(argv[i], "--rows")) {
            rows_list.clear();
            for (char* s = argv[i + 1]; *s;) { rows_list.push_back(std::strtol(s, &s, 10)); if (*s == ',') { ++s; } }
        }
    }
    int sms = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
    if (splits == 0) { splits = std::clamp(2 * sms / kKVH, 1, 128); } // executor.cpp's decode_splits_full
    if (near_splits == 0) { near_splits = std::min(splits, 64); }      // and its far-route defaults
    if (far_splits == 0) { far_splits = 128 - near_splits; }
    const KvCoding kc = kv == "rk8v4" ? KvCoding::Int8G64 : KvCoding::Lloyd4, vc = KvCoding::Int4G32;
    const int slots = (depth + 16 + 15) / 16 * 16;

    auto plane = [&](KvCoding c, unsigned seed) {
        KvPlaneView p;
        p.coding     = c;
        p.code_row   = kv_code_bytes(c);
        p.scale_row  = kv_scale_bytes(c);
        p.code_head  = p.code_row * slots;
        p.scale_head = p.scale_row * slots;
        p.code_seq   = p.code_head * kKVH;
        p.scale_seq  = p.scale_head * kKVH;
        p.codes      = static_cast<std::uint8_t*>(dev(p.code_seq));
        p.scales     = static_cast<std::uint8_t*>(dev(p.scale_seq));
        fill_bytes<<<1024, 256>>>(p.codes, p.code_seq, seed);
        fill_half<<<1024, 256>>>(reinterpret_cast<__half*>(p.scales), p.scale_seq / 2, c == KvCoding::Int4G32 ? 0.05f : 0.02f);
        return p;
    };
    const KvPlaneView kp = plane(kc, 1), vp = plane(vc, 2);
    const int tmax = std::max(chunk, 16);
    float* q     = static_cast<float*>(dev(std::size_t(tmax) * kQH * 128 * 4));
    float* r     = static_cast<float*>(dev(std::size_t(tmax) * kQH * kRel * 4));
    float* sinks = static_cast<float*>(dev(std::size_t(tmax) * kQH * 4));
    float* table = static_cast<float*>(dev(std::size_t(kExtent) * kRel * 4));
    float* gate  = static_cast<float*>(dev(std::size_t(tmax) * kQH * 128 * 4));
    auto* out    = static_cast<__nv_bfloat16*>(dev(std::size_t(tmax) * kQH * 128 * 2));
    auto* out2   = static_cast<__nv_bfloat16*>(dev(std::size_t(tmax) * kQH * 128 * 2));
    float* partial = static_cast<float*>(dev(k::attention_workspace_floats(16, kQH, 128) * 4));
    fill_float<<<1024, 256>>>(q, std::size_t(tmax) * kQH * 128, qscale, 3);
    fill_float<<<1024, 256>>>(r, std::size_t(tmax) * kQH * kRel, 0.5f, 4);
    fill_float<<<1024, 256>>>(sinks, std::size_t(tmax) * kQH, 1.0f, 5);
    fill_float<<<1024, 256>>>(table, std::size_t(kExtent) * kRel, 0.5f, 6);
    fill_float<<<1024, 256>>>(gate, std::size_t(tmax) * kQH * 128, 2.0f, 7);
    std::vector<std::int32_t> pos(tmax);
    std::int32_t* positions = static_cast<std::int32_t*>(dev(tmax * 4));
    CUDA_CHECK(cudaDeviceSynchronize());

    auto base_args = [&]() {
        k::AttentionArgs a;
        a.q = q; a.r = r; a.sinks = sinks; a.sink_stride = kQH; a.table = table; a.extent = kExtent;
        a.k_plane = kp; a.v_plane = vp; a.query_heads = kQH; a.kv_heads = kKVH;
        a.scale = 1.0f / std::sqrt(128.0f); a.gate = gate; a.gate_stride = kQH * 128; a.out = out;
        return a;
    };
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    auto time_call = [&](const k::AttentionArgs& a) {
        k::attention(a, nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaEventRecord(e0));
        for (int i = 0; i < reps; ++i) { k::attention(a, nullptr); }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        return ms / reps;
    };
    const double kv_bytes_per_key = kKVH * double(kv_code_bytes(kc) + kv_scale_bytes(kc) + kv_code_bytes(vc) + kv_scale_bytes(vc));
    std::printf("depth %d, kv %s, %d SMs, scalar splits %d, far route %d + %d, KV of one visit %.1f MiB\n", depth,
                kv.c_str(), sms, splits, near_splits, far_splits, kv_bytes_per_key * depth / 1048576.0);
    if (chunk > 0 && depth > chunk) {
        const int first = depth - chunk;
        std::vector<int> rows_pos(chunk);
        for (int t = 0; t < chunk; ++t) { pos[t] = rows_pos[t] = first + t; }
        CUDA_CHECK(cudaMemcpy(positions, pos.data(), chunk * 4, cudaMemcpyHostToDevice));
        k::AttentionArgs a = base_args();
        a.rows.positions = positions; a.rows.tokens = chunk; a.rows.chunk = true; a.rows.first = first;
        a.splits = 1;
        const float ms = time_call(a);
        const double flops = 4.0 * kQH * 128 * (double(first) * chunk + 0.5 * double(chunk) * chunk);
        std::printf("prompt chunk %d rows after %d keys: scalar %.2f ms per visit, x8 visits %.1f ms = %.0f tok/s attention only, %.1f TFLOPS\n",
                    chunk, first, ms, 8 * ms, chunk / (8 * ms / 1000.0), flops / (ms / 1000.0) / 1e12);
        if (chunk_far > 0) {
            k::AttentionArgs f = a;
            f.far_splits = chunk_far; f.out = out2;
            f.partial = static_cast<float*>(dev(k::attention_workspace_floats(chunk, kQH, 1 + chunk_far) * 4));
            const float mf = time_call(f);
            std::printf("prompt chunk far route (1 + %d slots): %.2f ms per visit, x8 visits %.1f ms = %.0f tok/s attention only, %.1f TFLOPS, %.2fx\n",
                        chunk_far, mf, 8 * mf, chunk / (8 * mf / 1000.0), flops / (mf / 1000.0) / 1e12, ms / mf);
            if (check) {
                const std::vector<double> ref = oracle(kp, vp, a, rows_pos);
                report("chunk scalar", out, ref);
                report("chunk far   ", out2, ref);
            }
            CUDA_CHECK(cudaFree(f.partial));
        }
    }
    for (int t : rows_list) {
        std::vector<int> rows_pos(t);
        for (int i = 0; i < t; ++i) { pos[i] = rows_pos[i] = depth - t + i; }
        CUDA_CHECK(cudaMemcpy(positions, pos.data(), t * 4, cudaMemcpyHostToDevice));
        k::AttentionArgs a = base_args();
        a.rows.positions = positions; a.rows.tokens = t; a.rows.chunk = false;
        a.splits = splits; a.partial = partial;
        const float ms = time_call(a);
        k::AttentionArgs f = a;
        f.splits = near_splits; f.far_splits = far_splits; f.out = out2;
        const float mf = time_call(f);
        std::printf("decode %d row(s): scalar %.3f ms per visit (x8 %.2f ms, %.0f GB/s); far route %.3f ms (x8 %.2f ms, %.0f GB/s), %.2fx\n",
                    t, ms, 8 * ms, kv_bytes_per_key * depth / (ms / 1000.0) / 1e9, mf, 8 * mf,
                    kv_bytes_per_key * depth / (mf / 1000.0) / 1e9, ms / mf);
        if (check) {
            const std::vector<double> ref = oracle(kp, vp, a, rows_pos);
            report("scalar", out, ref);
            report("far   ", out2, ref);
        }
    }
    return 0;
}
