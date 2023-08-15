//
// quantization
//

static void gptneox_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, enum gptneox_ftype ftype, int nthread) {
    ggml_type quantized_type;
    switch (ftype) {
        case GPTNEOX_FTYPE_MOSTLY_Q4_0: quantized_type = GGML_TYPE_Q4_0; break;
        case GPTNEOX_FTYPE_MOSTLY_Q4_1: quantized_type = GGML_TYPE_Q4_1; break;
        case GPTNEOX_FTYPE_MOSTLY_Q5_0: quantized_type = GGML_TYPE_Q5_0; break;
        case GPTNEOX_FTYPE_MOSTLY_Q5_1: quantized_type = GGML_TYPE_Q5_1; break;
        case GPTNEOX_FTYPE_MOSTLY_Q8_0: quantized_type = GGML_TYPE_Q8_0; break;
        default: throw format("invalid output file type %d\n", ftype);
    };

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    std::unique_ptr<gptneox_model_loader> model_loader(new gptneox_model_loader(fname_inp.c_str(), /*use_mmap*/ false,
                                                                            /*vocab_only*/ false));
    arch_util_file_saver file_saver(fname_out.c_str(), model_loader->file_loaders.at(0).get(), ftype);

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    std::vector<int64_t> hist_all(1 << 4, 0);

    std::vector<std::thread> workers;
    std::mutex mutex;

    size_t idx = 0;
    for (gptneox_load_tensor & tensor : model_loader->tensors_map.tensors) {
        arch_util_buffer read_data;
        read_data.resize(tensor.size);
        tensor.data = read_data.addr;
        model_loader->load_data_for(tensor);

        printf("[%4zu/%4zu] %36s - %16s, type = %6s, ",
               ++idx, model_loader->tensors_map.tensors.size(),
               tensor.name.c_str(), gptneox_format_tensor_shape(tensor.ne).c_str(),
               ggml_type_name(tensor.type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = tensor.name.rfind("weight") == tensor.name.size() - 6; // ends with 'weight'?

        // quantize only 2D tensors
        quantize &= (tensor.ne.size() == 2);

        // uncomment this to keep the output layer in FP16
        //if (tensor.name == "output.weight") {
        //    quantize = false;
        //}

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;
        arch_util_buffer work;

        if (!quantize) {
            new_type = tensor.type;
            new_data = tensor.data;
            new_size = tensor.size;
            printf("size = %8.3f MiB\n", tensor.size/1024.0/1024.0);
        } else {
            new_type = quantized_type;
            float * f32_data;
            size_t nelements = tensor.ne.at(0) * tensor.ne.at(1);
            arch_util_buffer f32_conv_buf;
            if (tensor.type == GGML_TYPE_F32) {
                f32_data = (float *) tensor.data;
            } else if (tensor.type == GGML_TYPE_F16) {
                f32_conv_buf.resize(nelements * sizeof(float));
                f32_data = (float *) f32_conv_buf.addr;
                auto f16_data = (const ggml_fp16_t *) tensor.data;
                for (size_t i = 0; i < nelements; i++) {
                    f32_data[i] = ggml_fp16_to_fp32(f16_data[i]);
                }
            } else {
                throw format("type %s unsupported for integer quantization", ggml_type_name(tensor.type));
            }

            printf("quantizing .. ");
            fflush(stdout);

            work.resize(nelements * 4); // upper bound on size
            new_data = work.addr;
            std::vector<int64_t> hist_cur(1 << 4, 0);

            int chunk_size = 32 * 512;
            const int nchunk = (nelements + chunk_size - 1)/chunk_size;
            const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
            if (nthread_use < 2) {
                new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nelements, hist_cur.data());
            } else {
                size_t counter = 0;
                new_size = 0;
                auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, nelements, chunk_size] () {
                    std::vector<int64_t> local_hist;
                    size_t local_size = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex);
                        size_t first = counter; counter += chunk_size;
                        if (first >= nelements) {
                            if (!local_hist.empty()) {
                                for (int j=0; j<int(local_hist.size()); ++j) hist_cur[j] += local_hist[j];
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        size_t last = std::min(nelements, first + chunk_size);
                        if (local_hist.empty()) local_hist.resize(hist_cur.size(), 0);
                        local_size += ggml_quantize_chunk(new_type, f32_data, new_data, first, last - first, local_hist.data());
                    }
                };
                if (int(workers.size()) < nthread_use - 1) workers.resize(nthread_use - 1);
                for (int it = 0; it < nthread_use - 1; ++it) workers[it] = std::thread(compute);
                compute();
                for (int it = 0; it < nthread_use - 1; ++it) workers[it].join();
            }

            printf("size = %8.2f MiB -> %8.2f MiB | hist: ", tensor.size/1024.0/1024.0, new_size/1024.0/1024.0);
            for (size_t i = 0; i < hist_cur.size(); i++) {
                hist_all[i] += hist_cur[i];
            }

            for (size_t i = 0; i < hist_cur.size(); i++) {
                printf("%5.3f ", hist_cur[i] / float(nelements));
            }
            printf("\n");
        }
        total_size_org += tensor.size;
        total_size_new += new_size;
        file_saver.write_tensor(tensor, new_type, new_data, new_size);
    }

    printf("%s: model size  = %8.2f MiB\n", __func__, total_size_org/1024.0/1024.0);
    printf("%s: quant size  = %8.2f MiB\n", __func__, total_size_new/1024.0/1024.0);

    {
        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); i++) {
            sum_all += hist_all[i];
        }

        printf("%s: hist: ", __func__);
        for (size_t i = 0; i < hist_all.size(); i++) {
            printf("%5.3f ", hist_all[i] / float(sum_all));
        }
        printf("\n");
    }
}

int gptneox_model_quantize(
        const char * fname_inp,
        const char * fname_out,
  enum gptneox_ftype   ftype,
        int          nthread) {
    try {
        gptneox_model_quantize_internal(fname_inp, fname_out, ftype, nthread);
        return 0;
    } catch (const std::string & err) {
        fprintf(stderr, "%s: failed to quantize: %s\n", __func__, err.c_str());
        return 1;
    }
}
