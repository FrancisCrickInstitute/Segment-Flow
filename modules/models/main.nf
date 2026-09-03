process computeImageIds {
    // Adds a stable image_id (plus a placeholder prep_hash) to every row of
    // the image CSV before any preprocessing branching
    conda "${moduleDir}/envs/conda_combine_stacks.yml"
    memory { 500.MB * task.attempt as MemoryUnit }
    time { 5.m * task.attempt }

    input:
    path img_csv

    output:
    path "with_ids_${img_csv}", emit: csv

    script:
    """
    python ${moduleDir}/resources/usr/bin/add_image_ids.py \
    --img-csv ${img_csv} \
    --output-csv with_ids_${img_csv}
    """
}

process preprocessImage {
    // Re-use the combine stacks conda env
    conda "${moduleDir}/envs/conda_combine_stacks.yml"
    memory { (Math.max((10.GB).toBytes(), image_path.size() * 2) * task.attempt) as MemoryUnit }
    time { 5.m * task.attempt }

    input:
    tuple val(meta), path(image_path), val(mask_fname)
    path img_csv

    output:
    // preprocess_image.py names both outputs by image_id
    path "${meta.image_id}.csv", emit: img_csv
    path "${meta.image_id}_*.ome.zarr", emit: prep_imgs
    // Log prep hashes and associated sets
    path "preprocess_hashes.txt", emit: hash_legend

    script:
    """
    echo '${groovy.json.JsonOutput.toJson(params.preprocess)}' > preprocess_params.json
    python ${moduleDir}/resources/usr/bin/preprocess_image.py \
    --img-path ${image_path} \
    --preprocess-params preprocess_params.json \
    --img-csv ${img_csv}
    """
}

process splitStacks {
    // Re-use the combine stacks conda env
    conda "${moduleDir}/envs/conda_combine_stacks.yml"
    memory { 500.MB * task.attempt as MemoryUnit }
    time { 5.m * task.attempt }
    // publishDir "$params.cache_dir", mode: 'copy'

    input:
    path csv_path
    path model_chkpt

    output:
    path "split_${csv_path}", emit: csv_file

    script:
    // Nextflow must have a string of comma separated values as input params, so split them here
    // https://github.com/nextflow-io/nextflow/issues/3595 should track this
    num_substacks = params.num_substacks.replace(",", " ")
    overlap = params.overlap.replace(",", " ")
    def mem_arg = (params.containsKey('memory_per_job') && params.memory_per_job) \
        ? "--memory-per-job ${(params.memory_per_job as MemoryUnit).toBytes()}" \
        : ""
    // Resolve the per-model compute cap (falling back to the global default), then apply the
    // per-deployment scale so weaker/stronger GPUs can tune all caps with one param.
    def cap = (params.model_max_substack instanceof Map ? params.model_max_substack[params.model] : null) \
        ?: params.max_substack
    def scale = (params.substack_scale ?: 1.0) as double
    // null on an axis = uncapped; scale only the integer caps
    def scaled_cap = cap.collect { it == null ? 'null' : Math.max(1, (it * scale) as int) }
    def cap_arg = "--max-substack ${scaled_cap.join(' ')}"
    """
    python ${moduleDir}/resources/usr/bin/create_splits.py \
    --img-csv ${csv_path} \
    --output-csv split_${csv_path} \
    --num-substacks $num_substacks \
    --overlap $overlap \
    $mem_arg \
    $cap_arg
    """
}

process downloadArtifact {
    // storeDir is the central AIoD cache. Nextflow checks whether the output
    // file already exists there before deciding to run this process:
    //   - Cache hit:  execution is skipped; the existing file is symlinked into
    //                 the task work directory
    //   - Cache miss: the script runs and the result is persisted to the store.
    // One process call per artifact means each has a single mandatory output, so
    // storeDir's cache check is always unambiguous (no optional outputs)
    conda "${moduleDir}/envs/conda_setup_model.yml"
    storeDir params.model_chkpt_dir

    input:
    tuple val(artifact_label), val(artifact_name), val(artifact_loc), val(artifact_type)

    output:
    tuple val(artifact_label), path("${artifact_name}"), emit: artifact

    script:
    """
    python ${moduleDir}/resources/usr/bin/download_model.py \
    --chkpt-loc  "${artifact_loc}" \
    --chkpt-type "${artifact_type}" \
    --chkpt-fname "${artifact_name}"
    """
}

process setupModel {
    // Queries the AIoD registry and writes one JSON metadata file per artifact
    // (checkpoint always present; config and finetuning only when the model has
    // them, i.e. nothing is emitted)
    conda "${moduleDir}/envs/conda_setup_model.yml"

    input:
    val model_name
    val model_version
    val model_task
    val user_config

    output:
    path "model_chkpt_meta.json",      emit: model_chkpt_meta
    path "model_config_meta.json",     emit: model_config_meta,     optional: true
    path "model_finetuning_meta.json", emit: model_finetuning_meta, optional: true
    path "model_meta.json",            emit: model_meta

    script:
    def userConfigArg = user_config ? "--user-config \"${user_config}\"" : ""
    """
    python ${moduleDir}/resources/usr/bin/setup_model.py \
    --model_name "${model_name}" \
    --model_version "${model_version}" \
    --task "${model_task}" \
    ${userConfigArg}
    """
}

process runModel {
    label 'gpu_process'
    conda "${moduleDir}/envs/${task.ext.condaDir}/conda_${params.model}.yml"
    // Symlink to where AIoD Napari plugin file watcher is looking
    publishDir "$mask_output_dir"

    input:
    tuple val(img_path_key), val(meta), val(mask_fname), val(idxs), path(image_path)
    val mask_output_dir
    path model_config
    path model_chkpt
    val model_type
    val output_mask_type
    val model_axes

    output:
    // Output mask_fname to uniquely group on image + preprocesing branch
    tuple val(mask_fname), val(meta), val(mask_output_dir), path("${mask_fname}_x${idxs[0]}-${idxs[1]}_y${idxs[2]}-${idxs[3]}_z${idxs[4]}-${idxs[5]}.rle"), emit: mask

    script:
    def modelAxesArg = model_axes ? "--model-axes \"${model_axes}\"" : ""
    """
    python ${moduleDir}/resources/usr/bin/run_${params.model}.py \
    --img-path ${image_path} \
    --mask-fname "${mask_fname}" \
    --output-dir "${mask_output_dir}" \
    --model-chkpt ${model_chkpt} \
    --model-type "${model_type}" \
    --model-config ${model_config} \
    --idxs ${idxs.join(" ")} \
    --channels ${meta.channels} \
    --num-slices ${meta.num_slices} \
    --output-mask-type ${output_mask_type} \
    ${modelAxesArg}
    """
}

process combineStacks {
    conda "${moduleDir}/envs/conda_combine_stacks.yml"
    // Add a minimum amount of memory, otherwise scale as a multiple of the input mask size
    // NOTE: Masks are RLE-compressed, so multiply by buffer (10) then by average compression factor (1000)
    memory { (Math.max((5.GB).toBytes(), masks*.size().sum() * 10000) * task.attempt) as MemoryUnit }
    // Give more base time if postprocessing
    time { params.postprocess ? 45.m * Math.pow(2, task.attempt) : 10.min * Math.pow(2, task.attempt) }
    publishDir "$mask_output_dir", mode: 'copy'

    input:
    tuple val(mask_fname), val(meta), val(model), val(mask_output_dir), path(masks, arity: '1..*')
    val postprocess
    val output_format
    val output_mask_type

    output:
    path("${mask_fname}_all.${output_format}")

    script:
    def postprocess = postprocess ? "--postprocess" : ""
    overlap = params.overlap.replace(",", " ")
    // Same run-level preprocess config preprocessImage receives - combine_stacks.py
    // matches meta.prep_hash against it to recover this branch's own preprocessing
    // set, rather than threading the raw set itself through the CSV pipeline.
    """
    echo ${task.memory}
    echo '${groovy.json.JsonOutput.toJson(params.preprocess)}' > preprocess_config.json
    python ${moduleDir}/resources/usr/bin/combine_stacks.py \
    --mask-fname "${mask_fname}" \
    --output-dir "${mask_output_dir}" \
    --masks ${masks} \
    --model ${model} \
    --image-size ${meta.num_slices} ${meta.height} ${meta.width} \
    --overlap $overlap \
    --iou-threshold ${params.iou_threshold} \
    --output-format ${output_format} \
    --output-mask-type ${output_mask_type} \
    --preprocess-config preprocess_config.json \
    --prep-hash "${meta.prep_hash ?: ""}" \
    ${postprocess}
    """
}
