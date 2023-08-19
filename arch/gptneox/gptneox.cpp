// Defines fileno on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstdint>
#include <cstdio>
#endif

#include "../arch-util.h"
#include "gptneox.h"

#include "../ggml.h"

#include <array>
#include <ctime>
#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <cstring>
#include <climits>
#include <memory>
#include <algorithm>
#include <numeric>
#include <initializer_list>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>

#define GPTNEOX_USE_SCRATCH
#define GPTNEOX_MAX_SCRATCH_BUFFERS 16

// available open-assistant based gptneox models
// OpenAssistant/stablelm-7b-sft-v7-epoch-3
// OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5
enum e_model {
    MODEL_UNKNOWN,
    MODEL_3B, // StabilityAI Base Alpha 3B
    MODEL_7B,
    MODEL_12B,
    MODEL_20B,
};

static const size_t MB = 1024*1024;

// computed for n_ctx == 2048
// TODO: dynamically determine these sizes
// TODO: To load the stablelm 3B model on my test XR will require some tricks, small ggml context size, mmap support, among others, but is maybe feasible, is a smaller n_ctx required? 512 instead of 2048/4096? Does mmap work as desired on iOS?
//       needs modifications in ggml

//
// ggml helpers
//

static void ggml_graph_compute_helper(std::vector<uint8_t> & buf, ggml_cgraph * graph, int n_threads) {
    struct ggml_cplan plan = ggml_graph_plan(graph, n_threads);

    if (plan.work_size > 0) {
        buf.resize(plan.work_size);
        plan.work_data = buf.data();
    }

    ggml_graph_compute(graph, &plan);
}

// TODO: Modify for gptneox, how are these values actually determined?
// TODO: This is now priority, 
static const std::map<e_model, size_t> & MEM_REQ_SCRATCH0()
{
    static std::map<e_model, size_t> _MEM_REQ_SCRATCH0 = {
        { MODEL_3B,    256ull * MB },
        { MODEL_7B,    512ull * MB },
        { MODEL_12B,   512ull * MB },
        { MODEL_20B,   512ull * MB },
    };
    return _MEM_REQ_SCRATCH0;
}

// TODO: Modify for gptneox, how are these values actually determined?
static const std::map<e_model, size_t> & MEM_REQ_SCRATCH1()
{
    static std::map<e_model, size_t> _MEM_REQ_SCRATCH1 = {
        { MODEL_3B,    256ull * MB },
        { MODEL_7B,    512ull * MB },
        { MODEL_12B,   512ull * MB },
        { MODEL_20B,   512ull * MB },
    };
    return _MEM_REQ_SCRATCH1;
}

// TODO: Modify for gptneox, how are these values actually determined?
// 2*n_embd*n_ctx*n_layer*sizeof(float16)
// llama 7B: 2 * 768 * 32 * 2 = 98304
static const std::map<e_model, size_t> & MEM_REQ_KV_SELF()
{
    static std::map<e_model, size_t> _MEM_REQ_KV_SELF = {
        { MODEL_3B,   512ull * MB },
        { MODEL_7B,   1026ull * MB },
        { MODEL_12B,  1608ull * MB },
        { MODEL_20B,  1608ull * MB },
    };
    return _MEM_REQ_KV_SELF;
}

// TODO: Modify for gptneox, how are these values actually determined?
// this is mostly needed for temporary mul_mat buffers to dequantize the data
// not actually needed if BLAS is disabled
static const std::map<e_model, size_t> & MEM_REQ_EVAL()
{
    static std::map<e_model, size_t> _MEM_REQ_EVAL = {
        { MODEL_3B,   512ull * MB },
        { MODEL_7B,   768ull * MB },
        { MODEL_12B, 1024ull * MB },
        { MODEL_20B, 1024ull * MB },
    };
    return _MEM_REQ_EVAL;
}

// default hparams (GPT-NeoX oasst 12B)
struct gptneox_hparams {
    uint32_t n_vocab = 50288;
    uint32_t n_ctx   = 4096;   // this is provided as user input?
    uint32_t n_embd  = 5120;
    uint32_t n_head  = 40;
    uint32_t n_layer = 36;
    uint32_t n_rot   = 32;
    uint32_t use_parallel_residual = 1; // 1 = true, 0 = false
    enum gptneox_ftype ftype = GPTNEOX_FTYPE_MOSTLY_F16;

    bool operator!=(const gptneox_hparams & other) const {
        return memcmp(this, &other, sizeof(gptneox_hparams));
    }
};

struct gptneox_layer {
    // input_layernorm
    struct ggml_tensor * ln_attn_g;
    struct ggml_tensor * ln_attn_b;

    // post_attention_layernorm
    struct ggml_tensor * ln_ff_g;
    struct ggml_tensor * ln_ff_b;

    // attention
    struct ggml_tensor * c_attn_attn_w;

    struct ggml_tensor * c_attn_attn_b;

    struct ggml_tensor * c_attn_proj_w;
    struct ggml_tensor * c_attn_proj_b;

    // ff
    struct ggml_tensor * c_mlp_fc_w;
    struct ggml_tensor * c_mlp_fc_b;

    struct ggml_tensor * c_mlp_proj_w;
    struct ggml_tensor * c_mlp_proj_b;
};

struct gptneox_kv_cache {
    struct ggml_tensor * k;
    struct ggml_tensor * v;

    struct ggml_context * ctx = NULL;

    arch_util_buffer buf;

    int n; // number of tokens currently in the cache

    ~gptneox_kv_cache() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct gptneox_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};

struct gptneox_model {
    e_model type = MODEL_UNKNOWN;

    gptneox_hparams hparams;

    // final normalization
    struct ggml_tensor * ln_f_g;
    struct ggml_tensor * ln_f_b;

    // word embedding
    struct ggml_tensor * wte;

    // language model head
    struct ggml_tensor * lmh_g;

    std::vector<gptneox_layer> layers;

    // context
    struct ggml_context * ctx = NULL;

    // the model memory buffer
    arch_util_buffer buf;

    // model memory mapped file
    std::unique_ptr<arch_util_mmap> mapping;

    // objects representing data potentially being locked in memory
    arch_util_mlock mlock_buf;
    arch_util_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    gptneox_vocab vocab;

    ~gptneox_model() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct gptneox_context {
    gptneox_context(const gptneox_model & model) : model(model), t_load_us(model.t_load_us), t_start_us(model.t_start_us) {}
    ~gptneox_context() {
        if (model_owner) {
            delete &model;
        }
    }

    std::mt19937 rng;

    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    const gptneox_model & model;

    bool model_owner = false;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;

    // key + value cache for the self attention
    struct gptneox_kv_cache kv_self;

    size_t mem_per_token = 0;

    // decode output (2-dimensional array: [n_tokens][n_vocab])
    std::vector<float> logits;
    bool logits_all = false;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // reusable buffer for `struct ggml_graph_plan.work_data`
    std::vector<uint8_t> work_buffer;

    // memory buffers used to evaluate the model
    // TODO: move in gptneox_state
    arch_util_buffer buf_compute;
    arch_util_buffer buf_scratch[GPTNEOX_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[GPTNEOX_MAX_SCRATCH_BUFFERS] = { 0 };

    void use_buf(struct ggml_context * ctx, int i) {
#if defined(GPTNEOX_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size, buf.addr, });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) const {
#if defined(GPTNEOX_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};

template <typename T>
static T checked_mul(T a, T b) {
    T ret = a * b;
    if (a != 0 && ret / a != b) {
        throw format("overflow multiplying %llu * %llu",
                     (unsigned long long) a, (unsigned long long) b);
    }
    return ret;
}

static size_t checked_div(size_t a, size_t b) {
    if (b == 0 || a % b != 0) {
        throw format("error dividing %zu / %zu", a, b);
    }
    return a / b;
}

static std::string gptneox_format_tensor_shape(const std::vector<uint32_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5u", ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " x %5u", ne.at(i));
    }
    return buf;
}

static size_t gptneox_calc_tensor_size(const std::vector<uint32_t> & ne, enum ggml_type type) {
    size_t size = ggml_type_size(type);
    for (uint32_t dim : ne) {
        size = checked_mul<size_t>(size, dim);
    }
    return size / ggml_blck_size(type);
}

struct gptneox_load_tensor {
    std::string name;
    enum ggml_type type = GGML_TYPE_F32;
    std::vector<uint32_t> ne;
    size_t file_off;
    size_t size;
    struct ggml_tensor * ggml_tensor = NULL;
    uint8_t * data;
};

struct gptneox_load_tensors_map {
    // tensors is kept in a separate vector to preserve file order
    std::vector<gptneox_load_tensor> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;
};

enum arch_util_file_version {
    GPTNEOX_FILE_VERSION_GGML,
    GPTNEOX_FILE_VERSION_GGMF_V1, // added version field and scores in vocab
    GPTNEOX_FILE_VERSION_GGJT_V1, // added padding
};

struct arch_util_file_loader {
    arch_util_file file;
    arch_util_file_version file_version;
    gptneox_hparams hparams;
    gptneox_vocab vocab;

    arch_util_file_loader(const char * fname, gptneox_load_tensors_map & tensors_map)
        : file(fname, "rb") {
        fprintf(stderr, "gptneox.cpp: loading model from %s\n", fname);
        read_magic();
        read_hparams();
        read_vocab();
        read_tensor_metadata(tensors_map);
    }
    void read_magic() {
        uint32_t magic = file.read_u32();
        uint32_t version = 0;

        if (magic != 'ggml') {
            version = file.read_u32();
        }

        if (magic == 'ggml' && version == 0) {
            file_version = GPTNEOX_FILE_VERSION_GGML;
        } else if (magic == 'ggmf' && version == 1) {
            file_version = GPTNEOX_FILE_VERSION_GGMF_V1;
        } else if (magic == 'ggjt' && version == 1) {
            file_version = GPTNEOX_FILE_VERSION_GGJT_V1;
        } else {
            throw format("unknown (magic, version) combination: %08x, %08x; is this really a GGML file?",
                         magic, version);
        }
    }
    void read_hparams() {
        hparams.n_vocab = file.read_u32();
        hparams.n_ctx = file.read_u32();
        hparams.n_embd = file.read_u32();
        hparams.n_head = file.read_u32();
        hparams.n_layer = file.read_u32();
        hparams.n_rot = file.read_u32();
        hparams.use_parallel_residual = file.read_u32();
        hparams.ftype = (enum gptneox_ftype) file.read_u32();
    }
    void read_vocab() {
        vocab.id_to_token.resize(hparams.n_vocab);

        for (uint32_t i = 0; i < hparams.n_vocab; i++) {
            uint32_t len = file.read_u32();
            std::string word = file.read_string(len);

            float score = 0.0f;
            if (file_version >= GPTNEOX_FILE_VERSION_GGMF_V1) {
                file.read_raw(&score, sizeof(score));
            }

            vocab.token_to_id[word] = i;

            auto & tok_score = vocab.id_to_token[i];
            tok_score.tok = std::move(word);
            tok_score.score = score;
        }
    }
    void read_tensor_metadata(gptneox_load_tensors_map & tensors_map) {
        while (file.tell() < file.size) {
            gptneox_load_tensor tensor;
            uint32_t n_dims = file.read_u32();
            uint32_t name_len = file.read_u32();
            tensor.type = (enum ggml_type) file.read_u32();
            tensor.ne.resize(n_dims);
            file.read_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * n_dims);
            std::string name = file.read_string(name_len);
            if (n_dims < 1 || n_dims > 2) {
                throw format("gptneox.cpp: tensor '%s' should not be %u-dimensional", name.c_str(), n_dims);
            }
            switch (tensor.type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    break;
                default: {
                    throw format("unrecognized tensor type %u\n", tensor.type);
                }
            }

            // skip to the next multiple of 32 bytes
            if (file_version >= GPTNEOX_FILE_VERSION_GGJT_V1) {
                file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
            }

            tensor.file_off = file.tell();
            tensor.name = name;
            tensor.size = gptneox_calc_tensor_size(tensor.ne, tensor.type);
            file.seek(tensor.size, SEEK_CUR);

            tensors_map.tensors.push_back(tensor);
            tensors_map.name_to_idx[name] = tensors_map.tensors.size() - 1;
        }
    }
};

struct arch_util_file_saver {
    arch_util_file file;
    arch_util_file_loader * any_file_loader;
    arch_util_file_saver(const char * fname, arch_util_file_loader * any_file_loader, enum gptneox_ftype new_ftype)
        : file(fname, "wb"), any_file_loader(any_file_loader) {
        fprintf(stderr, "gptneox.cpp: saving model to %s\n", fname);
        write_magic();
        write_hparams(new_ftype);
        write_vocab();
    }
    void write_magic() {
        file.write_u32('ggjt'); // magic
        file.write_u32(1); // version
    }
    void write_hparams(enum gptneox_ftype new_ftype) {
        const gptneox_hparams & hparams = any_file_loader->hparams;
        file.write_u32(hparams.n_vocab);
        file.write_u32(hparams.n_ctx);
        file.write_u32(hparams.n_embd);
        file.write_u32(hparams.n_head);
        file.write_u32(hparams.n_layer);
        file.write_u32(hparams.n_rot);
        file.write_u32(hparams.use_parallel_residual);
        file.write_u32(new_ftype);
    }
    void write_vocab() {
        if (any_file_loader->file_version == GPTNEOX_FILE_VERSION_GGML) {
            fprintf(stderr, "gptneox.cpp: WARNING: input is an old file that doesn't have scores; will add dummy scores\n");
        }
        uint32_t n_vocab = any_file_loader->hparams.n_vocab;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const auto & token_score = any_file_loader->vocab.id_to_token.at(i);
            file.write_u32((uint32_t) token_score.tok.size());
            file.write_raw(token_score.tok.data(), token_score.tok.size());
            file.write_raw(&token_score.score, sizeof(token_score.score));
        }
    }
    void write_tensor(gptneox_load_tensor & tensor, enum ggml_type new_type, const void * new_data, size_t new_size) {
        switch (new_type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
                break;
            default: ARCH_ASSERT(false);
        }
        file.write_u32((uint32_t) tensor.ne.size());
        file.write_u32((uint32_t) tensor.name.size());
        file.write_u32(new_type);
        file.write_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * tensor.ne.size());
        file.write_raw(tensor.name.data(), tensor.name.size());
        file.seek(-static_cast<ptrdiff_t>(file.tell()) & 31, SEEK_CUR);
        ARCH_ASSERT(new_size == gptneox_calc_tensor_size(tensor.ne, new_type));
        file.write_raw(new_data, new_size);
    }
};

struct gptneox_model_loader {
    std::unique_ptr<arch_util_file_loader> file_loader;
    gptneox_load_tensors_map tensors_map;
    bool use_mmap;
    size_t num_ggml_tensors_created = 0;
    struct ggml_context * ggml_ctx = NULL;
    std::unique_ptr<arch_util_mmap> mapping;

    gptneox_model_loader(const std::string & fname_base, bool use_mmap) {
        file_loader = std::unique_ptr<arch_util_file_loader>(new arch_util_file_loader(fname_base.c_str(), tensors_map));
        if (!arch_util_mmap::SUPPORTED) {
            use_mmap = false;
        }
        this->use_mmap = use_mmap;
    }

    void calc_sizes(size_t * ctx_size_p, size_t * mmapped_size_p) const {
        *ctx_size_p = *mmapped_size_p = 0;
        for (const gptneox_load_tensor & lt : tensors_map.tensors) {
            *ctx_size_p += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
            *(use_mmap ? mmapped_size_p : ctx_size_p) += lt.size;
        }
    }

    struct ggml_tensor * get_tensor(const std::string & name, std::vector<uint32_t> ne) {
        auto it = tensors_map.name_to_idx.find(name);
        if (it == tensors_map.name_to_idx.end()) {
            throw format("gptneox.cpp: tensor '%s' is missing from model", name.c_str());
        }
        gptneox_load_tensor & lt = tensors_map.tensors.at(it->second);
        if (lt.ne != ne) {
            throw format("gptneox.cpp: tensor '%s' has wrong shape; expected %s, got %s",
                         name.c_str(), gptneox_format_tensor_shape(ne).c_str(), gptneox_format_tensor_shape(lt.ne).c_str());
        }
        
#if 0
        printf("%48s - %14s, type = %4s\n",
               lt.name.c_str(),
               gptneox_format_tensor_shape(lt.ne).c_str(),
               ggml_type_name(lt.type));
#endif

        return get_tensor_for(lt);
    }

    struct ggml_tensor * get_tensor_for(gptneox_load_tensor & lt) {
        struct ggml_tensor * tensor;
        if (lt.ne.size() == 2) {
            tensor = ggml_new_tensor_2d(ggml_ctx, lt.type, lt.ne.at(0), lt.ne.at(1));
        } else {
            ARCH_ASSERT(lt.ne.size() == 1);
            tensor = ggml_new_tensor_1d(ggml_ctx, lt.type, lt.ne.at(0));
        }
        ARCH_ASSERT(lt.ggml_tensor == NULL); // if this fails, we called get_tensor twice on the same tensor
        lt.ggml_tensor = tensor;
        num_ggml_tensors_created++;
        return tensor;
    }

    void done_getting_tensors() {
        if (num_ggml_tensors_created != tensors_map.tensors.size()) {
            throw std::string("gptneox.cpp: file contained more tensors than expected");
        }
    }

    void load_all_data(gptneox_progress_callback progress_callback, void *  progress_callback_user_data, arch_util_mlock * lmlock) {
        size_t data_size = 0;
        size_t lock_size = 0;
        for (const gptneox_load_tensor & lt : tensors_map.tensors) {
            data_size += lt.size;
        }

        if (use_mmap) {
            mapping.reset(new arch_util_mmap(&file_loader->file));
            if (!lmlock) {
                // Don't call the callback since the actual loading will be lazy
                // and we can't measure it.
                progress_callback = NULL;
            }
            if (lmlock) {
                lmlock->init(mapping->addr);
            }
        }

        size_t done_size = 0;
        for (gptneox_load_tensor & lt : tensors_map.tensors) {
            if (progress_callback) {
                progress_callback((float) done_size / data_size, progress_callback_user_data);
            }
            ARCH_ASSERT(lt.ggml_tensor); // unused tensors should have been caught by load_data already
            lt.data = (uint8_t *) lt.ggml_tensor->data;

            load_data_for(lt);

            // case GGML_BACKEND_CPU:
            lt.ggml_tensor->data = lt.data;
            if (use_mmap && lmlock) {
                lock_size += lt.size;
                lmlock->grow_to(lock_size);
            }

            done_size += lt.size;
        }
    }

    void load_data_for(gptneox_load_tensor & lt) {
        if (use_mmap) {
            lt.data = (uint8_t *) mapping->addr + lt.file_off;
        } else {
            arch_util_file & file = file_loader->file;
            file.seek(lt.file_off, SEEK_SET);
            file.read_raw(lt.data, lt.size);
        }

        if (0) {
            print_checksum(lt);
        }
    }

    static void print_checksum(gptneox_load_tensor & lt) {
        uint32_t sum = 0;
        for (size_t i = 0; i < lt.size; i++) {
            uint8_t byte = lt.data[i];
            sum = byte + (sum << 6) + (sum << 16) - sum; // sdbm hash
        }
        fprintf(stderr, "%s checksum: %#08x (%s, size %zu)\n", lt.name.c_str(), sum,
                gptneox_format_tensor_shape(lt.ne).c_str(), lt.size);
    }

};


//
// kv cache
//

static bool kv_cache_init(
        const struct gptneox_hparams & hparams,
             struct gptneox_kv_cache & cache,
                           ggml_type   wtype,
                                 int   n_ctx) {
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;

    const int64_t n_mem      = (int64_t)n_layer*n_ctx;
    const int64_t n_elements = n_embd*n_mem;

    cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);

    struct ggml_init_params params;
    params.mem_size   = cache.buf.size;
    params.mem_buffer = cache.buf.addr;
    params.no_alloc   = false;

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);

    return true;
}

struct gptneox_context_params gptneox_context_default_params() {
    struct gptneox_context_params result = {
        /*.seed                        =*/ DEFAULT_SEED,
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 512,
        /*.f16_kv                      =*/ true,
        /*.logits_all                  =*/ false,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.embedding                   =*/ false,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
    };

    return result;
}

bool gptneox_mmap_supported() {
    return arch_util_mmap::SUPPORTED;
}

bool gptneox_mlock_supported() {
    return arch_util_mlock::SUPPORTED;
}

//
// model loading
//

static const char *arch_util_file_version_name(arch_util_file_version version) {
    switch (version) {
        case GPTNEOX_FILE_VERSION_GGML: return "'ggml' (old version with low tokenizer quality and no mmap support)";
        case GPTNEOX_FILE_VERSION_GGMF_V1: return "ggmf v1 (old version with no mmap support)";
        case GPTNEOX_FILE_VERSION_GGJT_V1: return "ggjt v1 (latest)";
        default: ARCH_ASSERT(false);
    }
}

static const char *gptneox_ftype_name(enum gptneox_ftype ftype) {
    switch (ftype) {
        case GPTNEOX_FTYPE_ALL_F32:     return "all F32";
        case GPTNEOX_FTYPE_MOSTLY_F16:  return "mostly F16";
        case GPTNEOX_FTYPE_MOSTLY_Q4_0: return "mostly Q4_0";
        case GPTNEOX_FTYPE_MOSTLY_Q4_1: return "mostly Q4_1";
        case GPTNEOX_FTYPE_MOSTLY_Q4_1_SOME_F16:
                                      return "mostly Q4_1, some F16";
        case GPTNEOX_FTYPE_MOSTLY_Q4_2: return "mostly Q4_2";
        //case GPTNEOX_FTYPE_MOSTLY_Q4_3: return "mostly Q4_3";
        case GPTNEOX_FTYPE_MOSTLY_Q5_0: return "mostly Q5_0";
        case GPTNEOX_FTYPE_MOSTLY_Q5_1: return "mostly Q5_1";
        case GPTNEOX_FTYPE_MOSTLY_Q8_0: return "mostly Q8_0";
        default:                      return "unknown, may not work";
    }
}

static const char *gptneox_model_type_name(e_model type) {
    switch (type) {
        case MODEL_3B: return "3B";
        case MODEL_7B: return "7B";
        case MODEL_12B: return "12B";
        case MODEL_20B: return "20B";
        case MODEL_UNKNOWN: return "UNKNOWN";
        default: ARCH_ASSERT(false);
    }
}

static void gptneox_model_load_internal(
        const std::string & fname,
        gptneox_model & model,
        gptneox_vocab & vocab,
        int n_ctx,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        gptneox_progress_callback progress_callback,
        void * progress_callback_user_data) {

    model.t_start_us = ggml_time_us();

    std::unique_ptr<gptneox_model_loader> ml(new gptneox_model_loader(fname, use_mmap));

    vocab = std::move(ml->file_loader->vocab);
    model.hparams = ml->file_loader->hparams;

    arch_util_file_version file_version = ml->file_loader->file_version;

    auto & hparams = model.hparams;
    
    {
        switch (hparams.n_layer) {
            case 16: {
                if (hparams.n_embd < 6144) {
                    model.type = e_model::MODEL_3B;
                } else {
                    model.type = e_model::MODEL_7B;
                }
                break;
            }
            case 32: {
                model.type = e_model::MODEL_7B;
                break;
            }
            case 36: model.type = e_model::MODEL_12B; break;
            case 44: model.type = e_model::MODEL_20B; break;
        }

        hparams.n_ctx = n_ctx;
    }

    {
        fprintf(stderr, "%s: format     = %s\n",  __func__, arch_util_file_version_name(file_version));
        fprintf(stderr, "%s: n_vocab    = %u\n",  __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_ctx      = %u\n",  __func__, hparams.n_ctx);
        fprintf(stderr, "%s: n_embd     = %u\n",  __func__, hparams.n_embd);
        fprintf(stderr, "%s: n_head     = %u\n",  __func__, hparams.n_head);
        fprintf(stderr, "%s: n_layer    = %u\n",  __func__, hparams.n_layer);
        fprintf(stderr, "%s: n_rot      = %u\n",  __func__, hparams.n_rot);
        fprintf(stderr, "%s: use_parallel_residual = %d\n", __func__, hparams.use_parallel_residual);
        fprintf(stderr, "%s: ftype      = %u (%s)\n", __func__, hparams.ftype, gptneox_ftype_name(hparams.ftype));
        fprintf(stderr, "%s: model size = %s\n",  __func__, gptneox_model_type_name(model.type));
    }

    if (vocab_only) {
        return;
    }

    auto & ctx = model.ctx;

    size_t ctx_size, mmapped_size;
    ml->calc_sizes(&ctx_size, &mmapped_size);
    fprintf(stderr, "%s: ggml ctx size = %6.2f KiB\n", __func__, ctx_size/1024.0);

    // print memory requirements
    {
        const size_t scale = memory_type == GGML_TYPE_F32 ? 2 : 1;

        // this is the total memory required to run the inference
        const size_t mem_required =
            ctx_size +
            mmapped_size +
            MEM_REQ_SCRATCH0().at(model.type) +
            MEM_REQ_SCRATCH1().at(model.type) +
            MEM_REQ_EVAL().at(model.type);

        // this is the memory required by one gptneox_state
        const size_t mem_required_state =
            scale*MEM_REQ_KV_SELF().at(model.type);

        fprintf(stderr, "%s: mem required  = %7.2f MB (+ %7.2f MB per state)\n", __func__,
                mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);
    }

    // create the ggml context
    {
        model.buf.resize(ctx_size);
        if (use_mlock) {
            model.mlock_buf.init(model.buf.addr);
            model.mlock_buf.grow_to(model.buf.size);
        }

        struct ggml_init_params params = {
            /*.mem_size   =*/ model.buf.size,
            /*.mem_buffer =*/ model.buf.addr,
            /*.no_alloc   =*/ ml->use_mmap,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            throw format("ggml_init() failed");
        }
    }

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const uint32_t n_embd  = hparams.n_embd;
        const uint32_t n_layer = hparams.n_layer;
        const uint32_t n_vocab = hparams.n_vocab;

        ml->ggml_ctx = ctx;

        model.wte       = ml->get_tensor("gpt_neox.embed_in.weight",            {n_embd, n_vocab});
        model.ln_f_g    = ml->get_tensor("gpt_neox.final_layer_norm.weight",    {n_embd});
        model.ln_f_b    = ml->get_tensor("gpt_neox.final_layer_norm.bias",      {n_embd});
        model.lmh_g     = ml->get_tensor("embed_out.weight",                    {n_embd, n_vocab});

        model.layers.resize(n_layer);
        for (uint32_t i = 0; i < n_layer; ++i) {
            auto & layer = model.layers[i];

            std::string layers_i = "gpt_neox.layers." + std::to_string(i);

            layer.ln_attn_g = ml->get_tensor(layers_i + ".input_layernorm.weight", {n_embd});
            layer.ln_attn_b = ml->get_tensor(layers_i + ".input_layernorm.bias", {n_embd});

            layer.c_attn_attn_w = ml->get_tensor(layers_i + ".attention.query_key_value.weight", {n_embd, n_embd * 3});
            layer.c_attn_attn_b = ml->get_tensor(layers_i + ".attention.query_key_value.bias", {n_embd * 3});
            layer.c_attn_proj_w = ml->get_tensor(layers_i + ".attention.dense.weight", {n_embd, n_embd});
            layer.c_attn_proj_b = ml->get_tensor(layers_i + ".attention.dense.bias", {n_embd});

            layer.ln_ff_g = ml->get_tensor(layers_i + ".post_attention_layernorm.weight", {n_embd});
            layer.ln_ff_b = ml->get_tensor(layers_i + ".post_attention_layernorm.bias", {n_embd});

            layer.c_mlp_fc_w =   ml->get_tensor(layers_i + ".mlp.dense_h_to_4h.weight", {n_embd,   n_embd * 4});
            layer.c_mlp_fc_b =   ml->get_tensor(layers_i + ".mlp.dense_h_to_4h.bias",   {n_embd * 4});
            layer.c_mlp_proj_w = ml->get_tensor(layers_i + ".mlp.dense_4h_to_h.weight", {n_embd * 4,   n_embd});
            layer.c_mlp_proj_b = ml->get_tensor(layers_i + ".mlp.dense_4h_to_h.bias",   {n_embd});
        }
    }

    ml->done_getting_tensors();

    // populate `tensors_by_name`
    for (gptneox_load_tensor & lt : ml->tensors_map.tensors) {
        model.tensors_by_name.emplace_back(lt.name, lt.ggml_tensor);
    }

    ml->load_all_data(progress_callback, progress_callback_user_data, use_mlock ? &model.mlock_mmap : NULL);

    model.mapping = std::move(ml->mapping);

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    model.t_load_us = ggml_time_us() - model.t_start_us;
}

static bool gptneox_model_load(
        const std::string & fname,
        gptneox_model & model,
        gptneox_vocab & vocab,
        int n_ctx,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        gptneox_progress_callback progress_callback,
        void *progress_callback_user_data) {
    try {
        gptneox_model_load_internal(fname, model, vocab, n_ctx, memory_type, use_mmap, use_mlock,
                                  vocab_only, progress_callback, progress_callback_user_data);
        return true;
    } catch (const std::string & err) {
        fprintf(stderr, "error loading model: %s\n", err.c_str());
        return false;
    }
}

// Helpers

static inline struct ggml_tensor * layer_norm(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * weight, struct ggml_tensor * bias) {
    // LayerNorm in RefinedWeb is `x = (x - mean(x)) / sqrt(variance(x) + 1e-5) * weight + bias`
    // Looks like ggml_norm does the first part, we only need to apply weight & bias.
    struct ggml_tensor * cur = ggml_norm(ctx, x);
    return ggml_add_inplace(ctx,
            ggml_mul_inplace(ctx,
                cur,
                ggml_repeat(ctx, weight, cur)),
            ggml_repeat(ctx, bias, cur));
}

static struct ggml_cgraph * gptneox_build_graph(
         gptneox_context & lctx,
     const gptneox_token * tokens,
           const float * embd,
                   int   n_tokens,
                   int   n_past) {

    ARCH_ASSERT((!tokens && embd) || (tokens && !embd));

    const int N = n_tokens;

    auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    auto & kv_self = lctx.kv_self;

    ARCH_ASSERT(!!kv_self.ctx);

    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx   = hparams.n_ctx;
    const int n_head  = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot   = hparams.n_rot;
    const int head_dim = n_embd / n_head;

    auto & mem_per_token = lctx.mem_per_token;
    auto & buf_compute   = lctx.buf_compute;

    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_compute.size,
        /*.mem_buffer =*/ buf_compute.addr,
        /*.no_alloc   =*/ false,
    };

    struct ggml_context * ctx0 = ggml_init(params);

    ggml_cgraph * gf = ggml_new_graph(ctx0);

    struct ggml_tensor * inpL;

    if (tokens) {
        struct ggml_tensor * inp_tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);

        memcpy(inp_tokens->data, tokens, N * ggml_element_size(inp_tokens));
        ggml_set_name(inp_tokens, "inp_tokens");

        inpL = ggml_get_rows(ctx0, model.wte, inp_tokens);
    } else {
#ifdef GGML_USE_MPI
        GGML_ASSERT(false && "not implemented");
#endif

        inpL = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_embd, N);

        memcpy(inpL->data, embd, N * n_embd * ggml_element_size(inpL));
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = model.layers[i];
        
        lctx.use_buf(ctx0, 0);

        // input norm
        struct ggml_tensor * cur = layer_norm(ctx0, inpL, layer.ln_attn_g, layer.ln_attn_b);

        // self-attention
        {
            // attn
            // [3*n_embd, n_embd] - model.layers[il].c_attn_attn_w
            // [3*n_embd,      1] - model.layers[il].c_attn_attn_b
            // [  n_embd,      N] - cur (in)
            // [3*n_embd,      N] - cur (out)
            //
            // cur = attn_w*cur + attn_b
            // [3*n_embd, N]
            {
                cur = ggml_mul_mat(ctx0, layer.c_attn_attn_w, cur);
                cur = ggml_add_inplace(ctx0,
                        cur,
                        ggml_repeat(ctx0, layer.c_attn_attn_b, cur));
            }
             
            // Split QKV and make contiguous
            struct ggml_tensor * Qcur = ggml_view_3d(ctx0, cur,
                                            head_dim,
                                            n_head,
                                            N,
                                            ggml_element_size(cur) * 3 * head_dim,
                                            ggml_element_size(cur) * 3 * n_embd,
                                            ggml_element_size(cur) * head_dim * 0);
            struct ggml_tensor * Kcur = ggml_view_3d(ctx0, cur,
                                            head_dim,
                                            n_head,
                                            N,
                                            ggml_element_size(cur) * 3 * head_dim,
                                            ggml_element_size(cur) * 3 * n_embd,
                                            ggml_element_size(cur) * head_dim * 1);
            struct ggml_tensor * Vcur = ggml_view_3d(ctx0, cur,
                                            head_dim,
                                            n_head,
                                            N,
                                            ggml_element_size(cur) * 3 * head_dim,
                                            ggml_element_size(cur) * 3 * n_embd,
                                            ggml_element_size(cur) * head_dim * 2);
            // TODO: Flatten without copying, or see if non-contiguous can be used for any of QKV.
            Qcur = ggml_cpy(ctx0, Qcur,
                        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_dim, n_head, N));
            Kcur = ggml_cpy(ctx0, Kcur,
                        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_dim, n_head, N));
            Vcur = ggml_cpy(ctx0, Vcur,
                        ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, head_dim, n_head, N));
            
            // MARK: gptneox RoPE Q and K, before cache
            // Bit 2 for gptneox style (2)
            // Bit 1 is zero for dont skip n_past +(0), use (2+1) = (3) if rope is applied to cache of k (after cache only)
            //Qcur = ggml_rope(ctx0, Qcur, n_past, n_rot, 2);
            //Kcur = ggml_rope(ctx0, Kcur, n_past, n_rot, 2); //3);

            // store key and value to memory, not required if prompt if only a single token (not practical or likely)
            //if (N >= 1) {
                // Each entry in kv_self has byte size of (ggml_element_size * n_embd * n_ctx * n_layer)
                Vcur = ggml_view_2d(ctx0, Vcur,
                            n_embd,
                            N,
                            ggml_element_size(Vcur) * n_embd,
                            0);
                Vcur = ggml_transpose(ctx0, Vcur);
            
                struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k,
                                            n_embd * N, // num elements in current context (up to n_embd*n_ctx but usually less)
                                            ggml_element_size(kv_self.k) * n_embd * (i * n_ctx + n_past));
                struct ggml_tensor * v = ggml_view_2d(ctx0, kv_self.v,
                                            N,
                                            n_embd,
                                            ggml_element_size(kv_self.v) * n_ctx,
                                            ggml_element_size(kv_self.v) * ((i * n_ctx * n_embd) + n_past));
            
                // important: storing RoPE-ed version of K in the KV cache!
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(gf, ggml_cpy(ctx0, Vcur, v));
            //}
            
            // Test RoPE after kv cache
            // for when we want to keep as much of the context as possible, we do not want to recalc kv weights
            // but positional encoding will change when old tokens are removed
            // Q is not cached so it is simply the same as the before version
            Qcur = ggml_rope_inplace(ctx0, Qcur, n_past, n_rot, 2, 0);
            // RoPE all in k cache
            // TODO: Should be able to replace view 1d and reshape 3d with a single view 3d
            // Do we need a larger scratch for this temp duplication?
            struct ggml_tensor * Kall = ggml_dup(ctx0,
                                            ggml_reshape_3d(ctx0,
                                                ggml_view_1d(ctx0, kv_self.k,
                                                    (n_past + N) * n_embd,
                                                    ggml_element_size(kv_self.k) * i * n_ctx * n_embd),
                                                head_dim, n_head, n_past + N));
            Kall = ggml_rope_inplace(ctx0, Kall, 0 /*n_past*/, n_rot, 2, 0); //3);
            
            // Q = Qcur.contiguous().view(head_dim, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor * Q =
                ggml_permute(ctx0,
                        Qcur,
                        0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor * K =
                ggml_permute(ctx0, Kall,
                        /*ggml_reshape_3d(ctx0,
                            ggml_view_1d(ctx0, kv_self.k,
                                (n_past + N) * n_embd,
                                ggml_element_size(kv_self.k) * i * n_ctx * n_embd),
                            head_dim, n_head, n_past + N),*/
                        0, 2, 1, 3);

            // K * Q
            // Will use internally ggml_compute_forward_mul_mat_f16_f32 because K is f16 (cache) and Q is f32 (from q4_0)
            // Outputs [N, N, H, B], so it seems like this is correct for "scores"
            // K is internally transposed by ggml_mul_mat
            struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor * KQ_scaled = ggml_scale_inplace(ctx0, KQ,
                                                ggml_new_f32(ctx0, 1.0f/sqrt(float(head_dim))));
            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor * KQ_masked = ggml_diag_mask_inf_inplace(ctx0, KQ_scaled, n_past);
            // KQ = soft_max(KQ_masked)
            struct ggml_tensor * KQ_soft_max = ggml_soft_max_inplace(ctx0, KQ_masked);
            
            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor * V_trans = ggml_view_3d(ctx0, kv_self.v,
                                                n_past + N,
                                                head_dim,
                                                n_head,
                                                ggml_element_size(kv_self.v) * n_ctx,
                                                ggml_element_size(kv_self.v) * n_ctx * n_embd/n_head,
                                                ggml_element_size(kv_self.v) * i * n_ctx * n_embd);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0, KQV_merged,
                        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (first weight)
            cur = ggml_mul_mat(ctx0, layer.c_attn_proj_w, cur);

            // projection (then bias)
            cur = ggml_add_inplace(ctx0, cur, ggml_repeat(ctx0, layer.c_attn_proj_b, cur));
        }

        lctx.use_buf(ctx0, 1);
        
        if (hparams.use_parallel_residual == 1) {
            //printf("use_parallel_residual == 1\n");
            // This is independent of the self-attention result, so it could be done in parallel to the self-attention
            struct ggml_tensor * outAttn = cur;
            // post attention layer norm
            cur = layer_norm(ctx0, inpL, layer.ln_ff_g, layer.ln_ff_b);
            // feed-forward network
            {
                // note here we pass inpFF instead of cur
                cur = ggml_mul_mat(ctx0, layer.c_mlp_fc_w, cur);
                cur = ggml_add_inplace(ctx0,
                            cur,
                            ggml_repeat(ctx0, layer.c_mlp_fc_b, cur));
                // GELU activation
                cur = ggml_gelu(ctx0, cur);
                // projection
                // cur = proj_w*inpFF + proj_b
                cur = ggml_mul_mat(ctx0, layer.c_mlp_proj_w, cur);
                cur = ggml_add_inplace(ctx0,
                            cur,
                            ggml_repeat(ctx0, layer.c_mlp_proj_b, cur));
            }
            //# pseudocode:
            //# x = x + attn(ln1(x)) + mlp(ln2(x))
            // inpL = inpL + outAttn + cur
            cur = ggml_add_inplace(ctx0, cur, outAttn);
            inpL = ggml_add_inplace(ctx0, inpL, cur);
        } else if (hparams.use_parallel_residual == 0) {
            //printf("use_parallel_residual == 0\n");
            // This takes the self-attention residual output as input to Feedforward
            struct ggml_tensor * outAttn = cur;
            inpL = ggml_add(ctx0, inpL, outAttn);
            // post attention layer norm
            cur = layer_norm(ctx0, inpL, layer.ln_ff_g, layer.ln_ff_b);
            // feed-forward network
            {
                // note here we pass inpFF instead of cur
                cur = ggml_mul_mat(ctx0, layer.c_mlp_fc_w, cur);
                cur = ggml_add_inplace(ctx0,
                               cur,
                               ggml_repeat(ctx0, layer.c_mlp_fc_b, cur));
                // GELU activation
                cur = ggml_gelu(ctx0, cur);
                // projection
                // cur = proj_w*inpFF + proj_b
                cur = ggml_mul_mat(ctx0, layer.c_mlp_proj_w, cur);
                cur = ggml_add_inplace(ctx0,
                               cur,
                               ggml_repeat(ctx0, layer.c_mlp_proj_b, cur));
            }
            //# pseudocode:
            //# x = x + attn(ln1(x)) (residual above as input to mlp)
            //# x = x + mlp(ln2(x)) (residual after mlp aka inpFF + cur)
            inpL = ggml_add_inplace(ctx0, inpL, cur);
        } else {
            printf("use_parallel_residual == %d\n", hparams.use_parallel_residual);
            assert(0);
        }
    }

    lctx.use_buf(ctx0, 0);

    // norm
    inpL = layer_norm(ctx0, inpL, model.ln_f_g, model.ln_f_b);
    ggml_set_name(inpL, "result_norm");

    // lm_head
    inpL = ggml_mul_mat(ctx0, model.lmh_g, inpL);
    ggml_set_name(inpL, "result_output");

    lctx.use_buf(ctx0, -1);

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(gf, inpL);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0)/N;
    }

#if 0
    printf("\n%s: used_mem = %.3f MB, scratch -- %.3f MB %.3f MB\n", __func__,
            ggml_used_mem(ctx0)/1024.0/1024.0,
            lctx.get_buf_max_mem(0)/1024.0/1024.0,
            lctx.get_buf_max_mem(1)/1024.0/1024.0);
#endif

    ggml_free(ctx0);

    return gf;
}

// evaluate the transformer
//
//   - lctx:      llama context
//   - tokens:    new batch of tokens to process
//   - embd       embeddings input
//   - n_tokens   number of tokens
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//
static bool gptneox_eval_internal(
        gptneox_context & lctx,
    const gptneox_token * tokens,
            const float * embd,
                    int   n_tokens,
                    int   n_past,
                    int   n_threads) {

    ARCH_ASSERT((!tokens && embd) || (tokens && !embd));

    const int64_t t_start_us = ggml_time_us();

    const int N = n_tokens;

    const auto & model   = lctx.model;
    const auto & hparams = model.hparams;

    const auto & kv_self = lctx.kv_self;

    ARCH_ASSERT(!!kv_self.ctx);

    const int64_t n_embd      = hparams.n_embd;
    const int64_t n_vocab     = hparams.n_vocab;

    ggml_cgraph * gf = gptneox_build_graph(lctx, tokens, embd, n_tokens, n_past);

    // for big prompts, if BLAS is enabled, it is better to use only one thread
    // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
    n_threads = N >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_cublas() ? 1 : n_threads;

    struct ggml_tensor * res = gf->nodes[gf->n_nodes - 1];
    struct ggml_tensor * embeddings = gf->nodes[gf->n_nodes - 2];

    ARCH_ASSERT(strcmp(res->name, "result_output") == 0);
    ARCH_ASSERT(strcmp(embeddings->name, "result_norm") == 0);

    ggml_graph_compute_helper(lctx.work_buffer, gf, n_threads);

    // update kv token count
    lctx.kv_self.n = n_past + N;

    //if (cgraph_fname) {
    //    ggml_graph_export(gf, cgraph_fname);
    //}

#ifdef GGML_PERF
    // print timing information per ggml operation (for debugging purposes)
    // requires GGML_PERF to be defined
    ggml_graph_print(&gf);
#endif

    // plot the computation graph in dot format (for debugging purposes)
    //if (n_past%100 == 0) {
    //    ggml_graph_dump_dot(&gf, NULL, "llama.dot");
    //}

    // extract logits
    {
        auto & logits_out = lctx.logits;

        if (lctx.logits_all) {
            logits_out.resize(n_vocab * N);
            memcpy(logits_out.data(), (float *) ggml_get_data(res), sizeof(float)*n_vocab*N);
        } else {
            // return result for just the last token
            logits_out.resize(n_vocab);
            memcpy(logits_out.data(), (float *) ggml_get_data(res) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
        }
    }

    // extract embeddings
    if (lctx.embedding.size()) {
        auto & embedding_out = lctx.embedding;

        embedding_out.resize(n_embd);
        memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
    }

    // measure the performance only for the single-token evals
    if (N == 1) {
        lctx.t_eval_us += ggml_time_us() - t_start_us;
        lctx.n_eval++;
    }
    else if (N > 1) {
        lctx.t_p_eval_us += ggml_time_us() - t_start_us;
        lctx.n_p_eval += N;
    }

    return true;
}

#include "gptneox-tokenize.cpp"

//
// sampling
//

void gptneox_sample_softmax(struct gptneox_context * ctx, gptneox_token_data_array * candidates) {
    assert(candidates->size > 0);

    const int64_t t_start_sample_us = ggml_time_us();

    // Sort the logits in descending order
    if (!candidates->sorted) {
        std::sort(candidates->data, candidates->data + candidates->size, [](const gptneox_token_data & a, const gptneox_token_data & b) {
            return a.logit > b.logit;
        });
        candidates->sorted = true;
    }

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void gptneox_sample_top_k(struct gptneox_context * ctx, gptneox_token_data_array * candidates, int k, size_t min_keep) {
    const int64_t t_start_sample_us = ggml_time_us();

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) candidates->size);

    // Sort scores in descending order
    if (!candidates->sorted) {
        auto comp = [](const gptneox_token_data & a, const gptneox_token_data & b) {
            return a.logit > b.logit;
        };
        if (k == (int) candidates->size) {
            std::sort(candidates->data, candidates->data + candidates->size, comp);
        } else {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        }
        candidates->sorted = true;
    }
    candidates->size = k;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void gptneox_sample_top_p(struct gptneox_context * ctx, gptneox_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    gptneox_sample_softmax(ctx, candidates);

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        cum_sum += candidates->data[i].p;

        // Check if the running sum is greater than p or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void gptneox_sample_tail_free(struct gptneox_context * ctx, gptneox_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    gptneox_sample_softmax(nullptr, candidates);

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);
    for (float & value : second_derivatives) {
        value /= second_derivatives_sum;
    }

    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}


void gptneox_sample_typical(struct gptneox_context * ctx, gptneox_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the softmax of logits and calculate entropy
    gptneox_sample_softmax(nullptr, candidates);

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        entropy += -candidates->data[i].p * logf(candidates->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < candidates->size; ++i) {
        float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<gptneox_token_data> new_candidates;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates.push_back(candidates->data[idx]);
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void gptneox_sample_temperature(struct gptneox_context * ctx, gptneox_token_data_array * candidates_p, float temp) {
    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].logit /= temp;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void gptneox_sample_repetition_penalty(struct gptneox_context * ctx, gptneox_token_data_array * candidates, gptneox_token * last_tokens, size_t last_tokens_size, float penalty) {
    if (last_tokens_size == 0 || penalty == 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates->size; ++i) {
        auto token_iter = std::find(last_tokens, last_tokens + last_tokens_size, candidates->data[i].id);
        if (token_iter == last_tokens + last_tokens_size) {
            continue;
        }

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (candidates->data[i].logit <= 0) {
            candidates->data[i].logit *= penalty;
        } else {
            candidates->data[i].logit /= penalty;
        }
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void gptneox_sample_frequency_and_presence_penalties(struct gptneox_context * ctx, gptneox_token_data_array * candidates, gptneox_token * last_tokens_p, size_t last_tokens_size, float alpha_frequency, float alpha_presence) {
    if (last_tokens_size == 0 || (alpha_frequency == 0.0f && alpha_presence == 0.0f)) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<gptneox_token, int> token_count;
    for (size_t i = 0; i < last_tokens_size; ++i) {
        token_count[last_tokens_p[i]]++;
    }

    // Apply frequency and presence penalties to the candidates
    for (size_t i = 0; i < candidates->size; ++i) {
        auto token_iter = token_count.find(candidates->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        int count = token_iter->second;
        candidates->data[i].logit -= float(count) * alpha_frequency + float(count > 0) * alpha_presence;
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}


gptneox_token gptneox_sample_token_mirostat(struct gptneox_context * ctx, gptneox_token_data_array * candidates, float tau, float eta, int m, float * mu) {
    assert(ctx);
    auto N = float(gptneox_n_vocab(ctx));
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    gptneox_sample_softmax(nullptr, candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    gptneox_sample_top_k(nullptr, candidates, int(k), 1);
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    gptneox_token X = gptneox_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const gptneox_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
        ctx->n_sample++;
    }
    return X;
}

gptneox_token gptneox_sample_token_mirostat_v2(struct gptneox_context * ctx, gptneox_token_data_array * candidates, float tau, float eta, float * mu) {
    assert(ctx);
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    gptneox_sample_softmax(ctx, candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const gptneox_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    // Normalize the probabilities of the remaining words
    gptneox_sample_softmax(ctx, candidates);

    // Sample the next word X from the remaining words
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    gptneox_token X = gptneox_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const gptneox_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

gptneox_token gptneox_sample_token_greedy(struct gptneox_context * ctx, gptneox_token_data_array * candidates) {
    const int64_t t_start_sample_us = ggml_time_us();

    // Find max element
    auto max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const gptneox_token_data & a, const gptneox_token_data & b) {
        return a.logit < b.logit;
    });

    gptneox_token result = max_iter->id;
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
        ctx->n_sample++;
    }
    return result;
}

gptneox_token gptneox_sample_token(struct gptneox_context * ctx, gptneox_token_data_array * candidates) {
    assert(ctx);
    const int64_t t_start_sample_us = ggml_time_us();
    gptneox_sample_softmax(nullptr, candidates);

    std::vector<float> probs;
    probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    auto & rng = ctx->rng;
    int idx = dist(rng);

    gptneox_token result = candidates->data[idx].id;

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;
    return result;
}

#include "gptneox-update.cpp"
#include "gptneox-quantize.cpp"

//
// interface implementation
//

struct gptneox_model * gptneox_load_model_from_file(
                             const char * path_model,
            struct gptneox_context_params   params) {
    ggml_time_init();

    gptneox_model * model = new gptneox_model;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (!gptneox_model_load(path_model, *model, model->vocab, params.n_ctx, memory_type,
                          params.use_mmap, params.use_mlock, params.vocab_only,
                          params.progress_callback, params.progress_callback_user_data)) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        delete model;
        return nullptr;
    }

    return model;
}

void gptneox_free_model(struct gptneox_model * model) {
    delete model;
}

struct gptneox_context * gptneox_new_context_with_model(
                 struct gptneox_model * model,
        struct gptneox_context_params   params) {

    if (!model) {
        return nullptr;
    }

    gptneox_context * ctx = new gptneox_context(*model);

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                *cur_percentage_p = percentage;
                fprintf(stderr, ".");
                if (percentage >= 100) {
                    fprintf(stderr, "\n");
                }
            }
        };
    }

    ctx->rng = std::mt19937(params.seed);
    ctx->logits_all = params.logits_all;

    ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

    // reserve memory for context buffers
    if (!params.vocab_only) {
        if (!kv_cache_init(ctx->model.hparams, ctx->kv_self, memory_type, ctx->model.hparams.n_ctx)) {
            fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
            gptneox_free(ctx);
            return nullptr;
        }

        {
            const size_t memory_size = ggml_nbytes(ctx->kv_self.k) + ggml_nbytes(ctx->kv_self.v);
            fprintf(stderr, "%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
        }

        const auto & hparams = ctx->model.hparams;

        // resized during inference
        if (params.logits_all) {
            ctx->logits.reserve(hparams.n_ctx*hparams.n_vocab);
        } else {
            ctx->logits.reserve(hparams.n_vocab);
        }

        if (params.embedding){
            ctx->embedding.resize(hparams.n_embd);
        }

        ctx->buf_compute.resize(MEM_REQ_EVAL().at(ctx->model.type) + ggml_graph_overhead());

#ifdef GPTNEOX_USE_SCRATCH
        ctx->buf_scratch[0].resize(MEM_REQ_SCRATCH0().at(ctx->model.type));
        ctx->buf_scratch[1].resize(MEM_REQ_SCRATCH1().at(ctx->model.type));
#endif
    }

    return ctx;
}

struct gptneox_context * gptneox_init_from_file(
                             const char * path_model,
            struct gptneox_context_params   params) {

    struct gptneox_model * model = gptneox_load_model_from_file(path_model, params);
    if (!model) {
        return nullptr;
    }
    struct gptneox_context * ctx = gptneox_new_context_with_model(model, params);
    ctx->model_owner = true;
    return ctx;
}

void gptneox_free(struct gptneox_context * ctx) {
    delete ctx;
}

int gptneox_apply_lora_from_file_internal(const struct gptneox_model & model, const char * path_lora, const char * path_base_model, int n_threads) {
    fprintf(stderr, "%s: applying lora adapter from '%s' - please wait ...\n", __func__, path_lora);

    const int64_t t_start_lora_us = ggml_time_us();

    auto fin = std::ifstream(path_lora, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_lora);
        return 1;
    }

    // verify magic and version
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 'ggla') {
            fprintf(stderr, "%s: bad file magic\n", __func__);
            return 1;
        }
        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != 1) {
            fprintf(stderr, "%s: unsupported file version\n", __func__ );
            return 1;
        }
    }

    int32_t lora_r;
    int32_t lora_alpha;
    fin.read((char *) &lora_r, sizeof(lora_r));
    fin.read((char *) &lora_alpha, sizeof(lora_alpha));
    float scaling = (float)lora_alpha / (float)lora_r;

    fprintf(stderr, "%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r, lora_alpha, scaling);


    // create a temporary ggml context to store the lora tensors
    // todo: calculate size from biggest possible tensor
    std::vector<uint8_t> lora_buf(1024ull * 1024ull * 1024ull);
    struct ggml_init_params params;
    params.mem_size   = lora_buf.size();
    params.mem_buffer = lora_buf.data();
    params.no_alloc   = false;

    ggml_context * lora_ctx = ggml_init(params);
    std::unordered_map<std::string, struct ggml_tensor *> lora_tensors;

    // create a name -> tensor map of the model to accelerate lookups
    std::unordered_map<std::string, struct ggml_tensor*> model_tensors;
    for (auto & kv: model.tensors_by_name) {
        model_tensors.insert(kv);
    }


    // load base model
    std::unique_ptr<gptneox_model_loader> model_loader;
    ggml_context * base_ctx = NULL;
    arch_util_buffer base_buf;
    if (path_base_model) {
        fprintf(stderr, "%s: loading base model from '%s'\n", __func__, path_base_model);
        model_loader.reset(new gptneox_model_loader(path_base_model, /*use_mmap*/ true));

        size_t ctx_size, mmapped_size;
        model_loader->calc_sizes(&ctx_size, &mmapped_size);
        base_buf.resize(ctx_size);

        ggml_init_params base_params;
        base_params.mem_size   = base_buf.size;
        base_params.mem_buffer = base_buf.addr;
        base_params.no_alloc   = model_loader->use_mmap;

        base_ctx = ggml_init(base_params);

        model_loader->ggml_ctx = base_ctx;

        // maybe this should in gptneox_model_loader
        if (model_loader->use_mmap) {
            model_loader->mapping.reset(new arch_util_mmap(&model_loader->file_loader->file, /* prefetch */ false));
        }
    }

    // read tensors and apply
    bool warned = false;
    int n_tensors = 0;

    std::vector<uint8_t> work_buffer;

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ftype;

        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fin.read(reinterpret_cast<char *>(&length), sizeof(length));
        fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));
        if (fin.eof()) {
            break;
        }

        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }

        std::string name(length, 0);
        fin.read(&name[0], length);

        // check for lora suffix and get the type of tensor
        const std::string lora_suffix = ".lora";
        size_t pos = name.rfind(lora_suffix);
        if (pos == std::string::npos) {
            fprintf(stderr, "%s: error: '%s' is not a lora tensor\n", __func__, name.c_str());
            return 1;
        }

        std::string lora_type = name.substr(pos + lora_suffix.length());
        std::string base_name = name;
        base_name.erase(pos);
        // fprintf(stderr, "%s: %s => %s (lora type %s) ", __func__, name.c_str(),base_name.c_str(), lora_type.c_str());

        if (model_tensors.find(base_name.data()) == model_tensors.end()) {
            fprintf(stderr, "%s: unknown tensor '%s' in lora adapter\n", __func__, name.data());
            return 1;
        }

        // create ggml tensor
        ggml_type wtype;
        switch (ftype) {
            case 0: wtype = GGML_TYPE_F32;  break;
            case 1: wtype = GGML_TYPE_F16;  break;
            default:
                    {
                        fprintf(stderr, "%s: invalid tensor data type '%d'\n",
                                __func__, ftype);
                        return false;
                    }
        }
        ggml_tensor* lora_tensor;
        if (n_dims == 2) {
            lora_tensor = ggml_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1]);
        }
        else {
            fprintf(stderr, "%s: unsupported tensor dimension %d\n", __func__, n_dims);
            return 1;
        }

        // load tensor data
        size_t offset = fin.tellg();
        size_t tensor_data_size = ggml_nbytes(lora_tensor);
        offset = (offset + 31) & -32;
        fin.seekg(offset);
        fin.read((char*)lora_tensor->data, tensor_data_size);

        lora_tensors[name] = lora_tensor;

        // check if we have both A and B tensors and apply
        if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
            lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {

            ggml_tensor * dest_t = model_tensors[base_name];
            ggml_tensor * base_t;
            if (model_loader) {
                // load from base model
                if (model_loader->tensors_map.name_to_idx.find(base_name) == model_loader->tensors_map.name_to_idx.end()) {
                    fprintf(stderr, "%s: error: tensor '%s' not found in base model\n", __func__, base_name.c_str());
                    return 1;
                }
                size_t idx = model_loader->tensors_map.name_to_idx[base_name];
                gptneox_load_tensor & lt = model_loader->tensors_map.tensors[idx];
                base_t = model_loader->get_tensor(base_name, { (uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1] });
                lt.data = (uint8_t *) lt.ggml_tensor->data;
                model_loader->load_data_for(lt);
                lt.ggml_tensor->data = lt.data;
            }
            else {
                base_t = dest_t;
            }

            if (ggml_is_quantized(base_t->type)) {
                if (!warned) {
                    fprintf(stderr, "%s: warning: using a lora adapter with a quantized model may result in poor quality, "
                                    "use a f16 or f32 base model with --lora-base\n", __func__);
                    warned = true;
                }
            }

            ggml_tensor * loraA = lora_tensors[base_name + ".loraA"];
            ggml_tensor * loraB = lora_tensors[base_name + ".loraB"];

            if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
                fprintf(stderr, "%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64 ");"
                               " are you sure that this adapter is for this model?\n", __func__, base_t->ne[0], loraA->ne[1]);
                return 1;
            }

            // w = w + BA*s
            ggml_tensor * BA = ggml_mul_mat(lora_ctx, loraA, loraB);

            if (scaling != 1.0f) {
                ggml_tensor * scale_tensor = ggml_new_f32(lora_ctx, scaling);
                BA = ggml_scale(lora_ctx, BA, scale_tensor);
            }

            ggml_tensor * r;
            if (base_t == dest_t) {
                r = ggml_add_inplace(lora_ctx, dest_t, BA);
            }
            else {
                r = ggml_add(lora_ctx, base_t, BA);
                r = ggml_cpy(lora_ctx, r, dest_t);
            }

            struct ggml_cgraph gf = ggml_build_forward(r);

            ggml_graph_compute_helper(work_buffer, &gf, n_threads);

            // we won't need these tensors again, reset the context to save memory
            ggml_free(lora_ctx);
            lora_ctx = ggml_init(params);
            lora_tensors.clear();

            n_tensors++;
            if (n_tensors % 4 == 0)
                fprintf(stderr, ".");
        }
    }

    // TODO: this should be in a destructor, it will leak on failure
    ggml_free(lora_ctx);
    if (base_ctx) {
        ggml_free(base_ctx);
    }

    const int64_t t_lora_us = ggml_time_us() - t_start_lora_us;
    fprintf(stderr, " done (%.2f ms)\n", t_lora_us / 1000.0);

    return 0;
}

int gptneox_apply_lora_from_file(struct gptneox_context * ctx, const char * path_lora, const char * path_base_model, int n_threads) {
    try {
        return gptneox_apply_lora_from_file_internal(ctx->model, path_lora, path_base_model, n_threads);
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return 1;
    }
}

int gptneox_model_apply_lora_from_file(const struct gptneox_model * model, const char * path_lora, const char * path_base_model, int n_threads) {
    try {
        return gptneox_apply_lora_from_file_internal(*model, path_lora, path_base_model, n_threads);
    } catch (const std::exception & err) {
        fprintf(stderr, "%s: failed to apply lora adapter: %s\n", __func__, err.what());
        return 1;
    }
}

int gptneox_get_kv_cache_token_count(struct gptneox_context * ctx) {
    return ctx->kv_self.n;
}

// Assumes contiguous data
void gptneox_shift_kv_cache(struct gptneox_context * ctx, int n) {
    auto & model = ctx->model;
    auto & kv_self = ctx->kv_self;
    auto & hparams = model.hparams;
    const int n_layer = hparams.n_layer;
    const int n_embd = hparams.n_embd;
    const int n_ctx = hparams.n_ctx;
    for(int il = 0; il < n_layer; il++) {
        // K: Embeddings are in regular order so moving them is easy as copying the memory
        {
            int elem_byte_size = ggml_element_size(kv_self.k);
            uint8_t * dst_ptr = ((uint8_t *)kv_self.k->data) + (elem_byte_size * n_embd * (il * n_ctx));
            uint8_t * src_ptr = ((uint8_t *)kv_self.k->data) + (elem_byte_size * n_embd * (il * n_ctx + n));
            memcpy(dst_ptr, src_ptr, elem_byte_size * n_embd * (n_ctx - n));
        }
        
        // V: Embeddings are transposed so each embedding element must be copied separately
        {
            int elem_byte_size = ggml_element_size(kv_self.v);
            for(int i = 0; i < n_embd; i++) {
                uint8_t * dst_ptr = ((uint8_t *)kv_self.v->data) + (elem_byte_size * (il * n_ctx * i));
                uint8_t * src_ptr = ((uint8_t *)kv_self.v->data) + (elem_byte_size * (il * n_ctx * i + n));
                memcpy(dst_ptr, src_ptr, elem_byte_size * (n_ctx - n));
            }
        }
    }
}

#define GPTNEOX_MAX_RNG_STATE 64*1024

void gptneox_set_rng_seed(struct gptneox_context * ctx, int seed) {
    if (seed <= 0) {
        seed = time(NULL);
    }
    ctx->rng.seed(seed);
}

// Returns the size of the state
size_t gptneox_get_state_size(struct gptneox_context * ctx) {
    // we don't know size of rng until we actually serialize it. so reserve more than enough memory for its serialized state.
    // for reference, std::mt19937(1337) serializes to 6701 bytes.
    const size_t s_rng_size        = sizeof(size_t);
    const size_t s_rng             = GPTNEOX_MAX_RNG_STATE;
    const size_t s_logits_capacity = sizeof(size_t);
    const size_t s_logits_size     = sizeof(size_t);
    const size_t s_logits          = ctx->logits.capacity() * sizeof(float);
    const size_t s_embedding_size  = sizeof(size_t);
    const size_t s_embedding       = ctx->embedding.size() * sizeof(float);
    const size_t s_kv_size         = sizeof(size_t);
    const size_t s_kv_ntok         = sizeof(int);
    const size_t s_kv              = ctx->kv_self.buf.size;

    const size_t s_total = (
        + s_rng_size
        + s_rng
        + s_logits_capacity
        + s_logits_size
        + s_logits
        + s_embedding_size
        + s_embedding
        + s_kv_size
        + s_kv_ntok
        + s_kv
    );

    return s_total;
}

// Copies the state to the specified destination address
size_t gptneox_copy_state_data(struct gptneox_context * ctx, uint8_t * dest) {
    uint8_t * out = dest;

    // copy rng
    {
        std::stringstream rng_ss;
        rng_ss << ctx->rng;

        const size_t rng_size = rng_ss.str().size();
        char rng_buf[GPTNEOX_MAX_RNG_STATE];

        memset(&rng_buf[0], 0, GPTNEOX_MAX_RNG_STATE);
        memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

        memcpy(out, &rng_size,   sizeof(rng_size));    out += sizeof(rng_size);
        memcpy(out, &rng_buf[0], GPTNEOX_MAX_RNG_STATE); out += GPTNEOX_MAX_RNG_STATE;
    }

    // copy logits
    {
        const size_t logits_cap  = ctx->logits.capacity();
        const size_t logits_size = ctx->logits.size();

        memcpy(out, &logits_cap,  sizeof(logits_cap));  out += sizeof(logits_cap);
        memcpy(out, &logits_size, sizeof(logits_size)); out += sizeof(logits_size);

        if (logits_size) {
            memcpy(out, ctx->logits.data(), logits_size * sizeof(float));
        }

        out += logits_cap * sizeof(float);
    }

    // copy embeddings
    {
        const size_t embedding_size = ctx->embedding.size();

        memcpy(out, &embedding_size, sizeof(embedding_size)); out += sizeof(embedding_size);

        if (embedding_size) {
            memcpy(out, ctx->embedding.data(), embedding_size * sizeof(float));
            out += embedding_size * sizeof(float);
        }
    }

    // copy kv cache
    {
        const size_t kv_size = ctx->kv_self.buf.size;
        const int    kv_ntok = gptneox_get_kv_cache_token_count(ctx);

        memcpy(out, &kv_size, sizeof(kv_size)); out += sizeof(kv_size);
        memcpy(out, &kv_ntok, sizeof(kv_ntok)); out += sizeof(kv_ntok);

        if (kv_size) {
            memcpy(out, ctx->kv_self.buf.addr, kv_size); out += kv_size;
        }
    }

    const size_t written  = out - dest;
    const size_t expected = gptneox_get_state_size(ctx);

    ARCH_ASSERT(written == expected);

    return written;
}

// Sets the state reading from the specified source address
size_t gptneox_set_state_data(struct gptneox_context * ctx, const uint8_t * src) {
    const uint8_t * in = src;

    // set rng
    {
        size_t rng_size;
        char   rng_buf[GPTNEOX_MAX_RNG_STATE];

        memcpy(&rng_size,   in, sizeof(rng_size));    in += sizeof(rng_size);
        memcpy(&rng_buf[0], in, GPTNEOX_MAX_RNG_STATE); in += GPTNEOX_MAX_RNG_STATE;

        std::stringstream rng_ss;
        rng_ss.str(std::string(&rng_buf[0], rng_size));
        rng_ss >> ctx->rng;

        ARCH_ASSERT(rng_ss.fail() == false);
    }

    // set logits
    {
        size_t logits_cap;
        size_t logits_size;

        memcpy(&logits_cap,  in, sizeof(logits_cap));  in += sizeof(logits_cap);
        memcpy(&logits_size, in, sizeof(logits_size)); in += sizeof(logits_size);

        ARCH_ASSERT(ctx->logits.capacity() == logits_cap);

        if (logits_size) {
            ctx->logits.resize(logits_size);
            memcpy(ctx->logits.data(), in, logits_size * sizeof(float));
        }

        in += logits_cap * sizeof(float);
    }

    // set embeddings
    {
        size_t embedding_size;

        memcpy(&embedding_size, in, sizeof(embedding_size)); in += sizeof(embedding_size);

        ARCH_ASSERT(ctx->embedding.capacity() == embedding_size);

        if (embedding_size) {
            memcpy(ctx->embedding.data(), in, embedding_size * sizeof(float));
            in += embedding_size * sizeof(float);
        }
    }

    // set kv cache
    {
        size_t kv_size;
        int kv_ntok;

        memcpy(&kv_size, in, sizeof(kv_size)); in += sizeof(kv_size);
        memcpy(&kv_ntok, in, sizeof(kv_ntok)); in += sizeof(kv_ntok);

        if (kv_size) {
            ARCH_ASSERT(ctx->kv_self.buf.size == kv_size);

            void * k_data = ctx->kv_self.k->data; // remember data pointers
            void * v_data = ctx->kv_self.v->data; // because their value is stored in buf and overwritten by memcpy

            memcpy(ctx->kv_self.buf.addr, in, kv_size); in += kv_size;

            ctx->kv_self.k->data = k_data; // restore correct data pointers
            ctx->kv_self.v->data = v_data;

        }

        ctx->kv_self.n = kv_ntok;
    }

    const size_t nread    = in - src;
    const size_t expected = gptneox_get_state_size(ctx);

    ARCH_ASSERT(nread == expected);

    return nread;
}

int gptneox_eval(
        struct gptneox_context * ctx,
           const gptneox_token * tokens,
                         int   n_tokens,
                         int   n_past,
                         int   n_threads) {
    if (!gptneox_eval_internal(*ctx, tokens, nullptr, n_tokens, n_past, n_threads)) {
        fprintf(stderr, "%s: failed to eval\n", __func__);
        return 1;
    }
    // get a more accurate load time, upon first eval
    if (!ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }
    return 0;
}

int gptneox_tokenize(
        struct gptneox_context * ctx,
                  const char * text,
                 gptneox_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    auto res = gptneox_tokenize(ctx->model.vocab, text, add_bos);

    if (n_max_tokens < (int) res.size()) {
        fprintf(stderr, "%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int gptneox_n_vocab_from_model(const struct gptneox_model * model) {
    return model->vocab.id_to_token.size();
}

int gptneox_n_ctx_from_model(const struct gptneox_model * model) {
    return model->hparams.n_ctx;
}

int gptneox_n_embd_from_model(const struct gptneox_model * model) {
    return model->hparams.n_embd;
}

int gptneox_n_vocab(const struct gptneox_context * ctx) {
    return ctx->model.vocab.id_to_token.size();
}

int gptneox_n_ctx(const struct gptneox_context * ctx) {
    return ctx->model.hparams.n_ctx;
}

int gptneox_n_embd(const struct gptneox_context * ctx) {
    return ctx->model.hparams.n_embd;
}

float * gptneox_get_logits(struct gptneox_context * ctx) {
    return ctx->logits.data();
}

float * gptneox_get_embeddings(struct gptneox_context * ctx) {
    return ctx->embedding.data();
}

const char * gptneox_token_to_str_with_model(const struct gptneox_model * model, gptneox_token token) {
    if (token >= gptneox_n_vocab_from_model(model)) {
        return nullptr;
    }

    return model->vocab.id_to_token[token].tok.c_str();
}

const char * gptneox_token_to_str(const struct gptneox_context * ctx, gptneox_token token) {
    return gptneox_token_to_str_with_model(&ctx->model, token);
}

gptneox_token gptneox_str_to_token_with_model(const struct gptneox_model * model, const char * str) {
    auto token_iter = model->vocab.token_to_id.find(str);
    if (token_iter == model->vocab.token_to_id.end()) {
        return 0;
    }
    return token_iter->second;
}

gptneox_token gptneox_str_to_token(const struct gptneox_context * ctx, const char * str) {
    return gptneox_str_to_token_with_model(&ctx->model, str);
}

gptneox_token gptneox_token_bos() {
    return 0;
}

gptneox_token gptneox_token_eos() {
    return 0;
}


void gptneox_print_timings(struct gptneox_context * ctx) {
    const int64_t t_end_us = ggml_time_us();

    const int32_t n_sample = std::max(1, ctx->n_sample);
    const int32_t n_eval   = std::max(1, ctx->n_eval);
    const int32_t n_p_eval = std::max(1, ctx->n_p_eval);

    fprintf(stderr, "\n");
    fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0);
    fprintf(stderr, "%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * ctx->t_sample_us, n_sample, 1e-3 * ctx->t_sample_us / n_sample);
    fprintf(stderr, "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n", __func__, 1e-3 * ctx->t_p_eval_us, n_p_eval, 1e-3 * ctx->t_p_eval_us / n_p_eval);
    fprintf(stderr, "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * ctx->t_eval_us,   n_eval,   1e-3 * ctx->t_eval_us   / n_eval);
    fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us)/1000.0);
}

void gptneox_reset_timings(struct gptneox_context * ctx) {
    ctx->t_start_us = ggml_time_us();
    ctx->t_sample_us = ctx->n_sample = 0;
    ctx->t_eval_us   = ctx->n_eval   = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char * gptneox_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "         + std::to_string(ggml_cpu_has_avx())         + " | ";
    s += "AVX2 = "        + std::to_string(ggml_cpu_has_avx2())        + " | ";
    s += "AVX512 = "      + std::to_string(ggml_cpu_has_avx512())      + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_cpu_has_avx512_vnni()) + " | ";
    s += "FMA = "         + std::to_string(ggml_cpu_has_fma())         + " | ";
    s += "NEON = "        + std::to_string(ggml_cpu_has_neon())        + " | ";
    s += "ARM_FMA = "     + std::to_string(ggml_cpu_has_arm_fma())     + " | ";
    s += "F16C = "        + std::to_string(ggml_cpu_has_f16c())        + " | ";
    s += "FP16_VA = "     + std::to_string(ggml_cpu_has_fp16_va())     + " | ";
    s += "WASM_SIMD = "   + std::to_string(ggml_cpu_has_wasm_simd())   + " | ";
    s += "BLAS = "        + std::to_string(ggml_cpu_has_blas())        + " | ";
    s += "SSE3 = "        + std::to_string(ggml_cpu_has_sse3())        + " | ";
    s += "VSX = "         + std::to_string(ggml_cpu_has_vsx())         + " | ";

    return s.c_str();
}

// For internal test use
const std::vector<std::pair<std::string, struct ggml_tensor *>>& gptneox_internal_get_tensor_map(struct gptneox_context * ctx) {
    return ctx->model.tensors_by_name;
}

size_t gptneox_load_session_file(struct gptneox_context * ctx, const char * path_session, gptneox_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    // TODO leverage mmap
    arch_util_file file(path_session, "rb");
    const uint32_t magic = file.read_u32();
    const uint32_t version = file.read_u32();

    if (!(magic == 'ggsn' && version == 0)) {
        fprintf(stderr, "%s : unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
        return 0;
    }

    gptneox_hparams session_hparams;
    file.read_raw(&session_hparams, sizeof(gptneox_hparams));

    // REVIEW
    if (session_hparams != ctx->model.hparams) {
        fprintf(stderr, "%s : model hparams didn't match from session file!\n", __func__);
        return 0;
    }

    const uint32_t n_token_count = file.read_u32();
    ARCH_ASSERT(n_token_capacity >= n_token_count);
    file.read_raw(tokens_out, sizeof(gptneox_token) * n_token_count);
    *n_token_count_out = n_token_count;

    const size_t n_state_size = file.size - file.tell();
    const size_t n_orig_state_size = gptneox_get_state_size(ctx);
    if (n_state_size != n_orig_state_size) {
        fprintf(stderr, "%s : failed to validate state size\n", __func__);
    }
    std::unique_ptr<uint8_t[]> state_data(new uint8_t[n_state_size]);
    file.read_raw(state_data.get(), n_state_size);
    return gptneox_set_state_data(ctx, state_data.get());
}

size_t gptneox_save_session_file(struct gptneox_context * ctx, const char * path_session, const gptneox_token * tokens, size_t n_token_count) {
    // TODO save temp & swap
    arch_util_file file(path_session, "wb");

    const size_t n_state_size = gptneox_get_state_size(ctx);
    std::unique_ptr<uint8_t[]> state_data(new uint8_t[n_state_size]);
    gptneox_copy_state_data(ctx, state_data.get());

    file.write_u32('ggsn'); // magic
    file.write_u32(0); // version
    file.write_raw(&ctx->model.hparams, sizeof(gptneox_hparams));

    file.write_u32((uint32_t) n_token_count); // REVIEW
    file.write_raw(tokens, sizeof(gptneox_token) * n_token_count);

    file.write_raw(state_data.get(), n_state_size);
    return n_state_size; // REVIEW
}

