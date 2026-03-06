process preprocessImage {
    // Re-use the combine stacks conda env
    conda "${moduleDir}/envs/conda_combine_stacks.yml"
    memory { (Math.max((10.GB).toBytes(), image_path.size() * 2) * task.attempt) as MemoryUnit }
    time { 5.m * task.attempt }

    input:
    tuple val(meta), path(image_path), val(mask_fname)
    path img_csv

    output:
    path "${image_path.simpleName}.csv", emit: img_csv
    path "${image_path.simpleName}_*.${image_path.extension}", emit: prep_imgs

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
    path "${csv_path}", emit: csv_file

    script:
    // Nextflow must have a string of comma separated values as input params, so split them here
    // https://github.com/nextflow-io/nextflow/issues/3595 should track this
    num_substacks = params.num_substacks.replace(",", " ")
    overlap = params.overlap.replace(",", " ")
    """
    python ${moduleDir}/resources/usr/bin/create_splits.py \
    --img-csv ${csv_path} \
    --output-csv ${csv_path} \
    --num-substacks $num_substacks \
    --overlap $overlap
    """
}

process setupModel {
    conda "${moduleDir}/envs/conda_setup_model.yml"
    storeDir "$params.model_chkpt_dir"

    input:
    val model_name
    val model_version
    val model_task

    output:
    path "${model_version}_${model_task}.*", emit: model_chkpt
    path "${model_version}_${model_task}_config.*", emit: model_config
    // path "${model_version}_finetuning_meta.*", emit: finetuning_mrue

    script:
    """
    python ${moduleDir}/resources/usr/bin/setup_model.py \
    --model-name "${model_name}" \
    --model-version "${model_version}" \
    --task "${model_task}" \
    --cache-loc "$params.model_chkpt_dir"
    """
}

process runModel {
    label 'small_gpu'
    conda "${moduleDir}/envs/${task.ext.condaDir}/conda_${params.model}.yml"
    // Symlink to where AIoD Napari plugin file watcher is looking
    publishDir "$mask_output_dir"

    input:
    tuple val(image_name), val(meta), val(mask_fname), val(idxs), path(image_path)
    val mask_output_dir
    path model_config
    path model_chkpt
    val model_type

    output:
    tuple val("${image_path.baseName}"), val(meta), val(mask_fname), val(mask_output_dir), path("${mask_fname}_x${idxs[0]}-${idxs[1]}_y${idxs[2]}-${idxs[3]}_z${idxs[4]}-${idxs[5]}.rle"), emit: mask

    script:
    """
    python ${moduleDir}/resources/usr/bin/run_${params.model}.py \
    --img-path ${image_path} \
    --mask-fname "${mask_fname}" \
    --output-dir ${mask_output_dir} \
    --model-chkpt ${model_chkpt} \
    --model-type ${model_type} \
    --model-config ${model_config} \
    --idxs ${idxs.join(" ")} \
    --channels ${meta.channels} \
    --num-slices ${meta.num_slices}
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
    tuple val(img_simplename), val(meta), val(model), val(mask_fname), val(mask_output_dir), path(masks, arity: '1..*')
    val postprocess

    output:
    path("${mask_fname}_all.rle")

    script:
    def postprocess = postprocess ? "--postprocess" : ""
    overlap = params.overlap.replace(",", " ")
    """
    echo ${task.memory}
    python ${moduleDir}/resources/usr/bin/combine_stacks.py \
    --mask-fname "${mask_fname}" \
    --output-dir ${mask_output_dir} \
    --masks ${masks} \
    --model ${model} \
    --image-size ${meta.num_slices} ${meta.height} ${meta.width} \
    --overlap $overlap \
    --iou-threshold ${params.iou_threshold} \
    ${postprocess}
    """
}
