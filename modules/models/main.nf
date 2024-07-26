process splitStacks {
    // Re-use the combine stacks conda env
    conda "${moduleDir}/envs/conda_combine_stacks.yml"
    memory { 500.MB * task.attempt as MemoryUnit }
    time { 5.m * task.attempt }
    publishDir "$params.cache_dir", mode: 'copy'

    input:
    path csv_path

    output:
    path "${csv_path.getName()}", emit: csv_file

    script:
    // Nextflow must have a string of comma separated values as input params, so split them here
    // https://github.com/nextflow-io/nextflow/issues/3595 should track this
    num_substacks = params.num_substacks.replace(",", " ")
    overlap = params.overlap.replace(",", " ")
    """
    python ${moduleDir}/resources/usr/bin/create_splits.py \
    --img-csv ${params.img_dir} \
    --output-csv ${csv_path.getName()} \
    --num-substacks $num_substacks \
    --overlap $overlap
    """
}

process downloadModel {
    conda "${moduleDir}/envs/conda_download_model.yml"
    publishDir "$params.model_chkpt_dir", mode: 'copy'

    input:
    val model_chkpt_path
    val model_chkpt_loc
    val model_chkpt_type
    val model_chkpt_fname

    output:
    // This is where publishDir needs to come in, symlinking to chkpt repo
    path "${model_chkpt_fname}", emit: model_chkpt

    script:
    """
    python ${moduleDir}/resources/usr/bin/download_model.py \
    --chkpt-output-dir "$params.model_chkpt_dir" \
    --chkpt-loc ${model_chkpt_loc} \
    --chkpt-type ${model_chkpt_type} \
    --chkpt-fname "${model_chkpt_fname}"
    """
}

process runSAM {
    label 'small_gpu'
    conda "${moduleDir}/envs/${task.ext.condaDir}/conda_sam.yml"
    // Switch this to use publishDir and avoid path manipulation in python?

    input:
    tuple val(meta), path(image_path), val(mask_fname), val(idxs)
    val mask_output_dir
    path model_config
    path model_chkpt
    val model_type

    output:
    tuple val("${image_path.simpleName}"), val(meta), val(mask_fname), val(mask_output_dir), val("${mask_output_dir}/${mask_fname}_x${idxs[0]}-${idxs[1]}_y${idxs[2]}-${idxs[3]}_z${idxs[4]}-${idxs[5]}.npy"), emit: mask

    script:
    """
    python ${moduleDir}/resources/usr/bin/run_sam.py \
    --img-path ${image_path} \
    --mask-fname "${mask_fname}" \
    --output-dir ${mask_output_dir} \
    --model-chkpt ${model_chkpt} \
    --model-type ${model_type} \
    --model-config ${model_config} \
    --idxs ${idxs.join(" ")}
    """
}

process runUNET {
    label 'small_gpu'
    conda "${moduleDir}/envs/${task.ext.condaDir}/conda_unet.yml"

    input:
    tuple val(meta), path(image_path), val(mask_fname), val(idxs)
    val mask_output_dir
    path model_config
    path model_chkpt

    output:
    tuple val("${image_path.simpleName}"), val(meta), val(mask_fname), val(mask_output_dir), val("${mask_output_dir}/${mask_fname}_x${idxs[0]}-${idxs[1]}_y${idxs[2]}-${idxs[3]}_z${idxs[4]}-${idxs[5]}.npy"), emit: mask

    script:
    """
    echo ${moduleDir}/envs/${task.ext.condaDir}/conda_unet.yml
    python ${moduleDir}/resources/usr/bin/run_unet.py \
    --img-path ${image_path} \
    --mask-fname "${mask_fname}" \
    --output-dir ${mask_output_dir} \
    --model-chkpt ${model_chkpt} \
    --model-config ${model_config} \
    --idxs ${idxs.join(" ")}
    """
}

process runMITONET {
    label 'small_gpu'
    // time { 1.m * params.stack_size * params.img_size_tier * task.attempt }
    conda "${moduleDir}/envs/${task.ext.condaDir}/conda_mitonet.yml"

    input:
    tuple val(meta), path(image_path), val(mask_fname), val(idxs)
    val mask_output_dir
    path model_config
    path model_chkpt
    val model_type

    output:
    tuple val("${image_path.simpleName}"), val(meta), val(mask_fname), val(mask_output_dir), val("${mask_output_dir}/${mask_fname}_x${idxs[0]}-${idxs[1]}_y${idxs[2]}-${idxs[3]}_z${idxs[4]}-${idxs[5]}.npy"), emit: mask

    script:
    """
    python ${moduleDir}/resources/usr/bin/run_mitonet.py \
    --img-path ${image_path} \
    --mask-fname "${mask_fname}" \
    --output-dir ${mask_output_dir} \
    --model-chkpt ${model_chkpt} \
    --model-type ${model_type} \
    --model-config ${model_config} \
    --idxs ${idxs.join(" ")}
    """
}

process combineStacks {
    conda "${moduleDir}/envs/conda_combine_stacks.yml"
    // Add a minimum amount of memory, otherwise scale as a multiple of the input mask size
    memory { Math.max((300.MB).toBytes(), masks*.size().sum() * 5 * task.attempt) as MemoryUnit }
    // Give more base time if postprocessing
    time { params.postprocess ? 45.m * Math.pow(2, task.attempt) : 10.min * Math.pow(2, task.attempt) }

    input:
    tuple val(img_simplename), val(meta), val(model), val(mask_fname), val(mask_output_dir), path(masks, arity: '1..*')
    val postprocess

    output:
    stdout

    script:
    def postprocess = postprocess ? "--postprocess" : ""
    overlap = params.overlap.replace(",", " ")
    """
    echo ${task.memory}
    echo ${masks*.size().sum() as MemoryUnit}
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