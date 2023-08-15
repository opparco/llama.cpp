//
// updating
//

static void gptneox_model_update_internal(const std::string & fname_inp, const std::string & fname_out) {
    std::unique_ptr<gptneox_model_loader> model_loader(new gptneox_model_loader(fname_inp.c_str(),
                                                            /*use_mmap*/ false,
                                                            /*vocab_only*/ false));
    // Simply use the ftype of the first file
    auto ftype = model_loader->file_loaders[0]->hparams.ftype;
    arch_util_file_saver file_saver(fname_out.c_str(), model_loader->file_loaders.at(0).get(), ftype);

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

        file_saver.write_tensor(tensor, tensor.type, tensor.data, tensor.size);
    }
}

int gptneox_model_update(
        const char * fname_inp,
        const char * fname_out) {
    try {
        gptneox_model_update_internal(fname_inp, fname_out);
        return 0;
    } catch (const std::string & err) {
        fprintf(stderr, "%s: failed to copy: %s\n", __func__, err.c_str());
        return 1;
    }
}
