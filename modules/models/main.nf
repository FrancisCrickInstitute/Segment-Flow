
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
    --chkpt-path ${model_chkpt_path} \
    --chkpt-loc ${model_chkpt_loc} \
    --chkpt-type ${model_chkpt_type} \
    --chkpt-fname ${model_chkpt_fname}
    """
}

process runSAM {
    label 'small_gpu'
    conda "${moduleDir}/envs/conda_sam.yml"
    // Switch this to use publishDir and avoid path manipulation in python?

    input:
    tuple val(meta), path(image_path), val(mask_fname), val(start_idx), val(end_idx)
    val mask_output_dir
    path model_config
    path model_chkpt
    val model_type

    output:
    tuple val("${image_path.simpleName}"), val(mask_fname), val(mask_output_dir), val("${mask_output_dir}/${mask_fname}_${end_idx-start_idx}_${start_idx}.npy"), emit: mask

    script:
    """
    python ${moduleDir}/resources/usr/bin/run_sam.py \
    --img-path ${image_path} \
    --mask-fname ${mask_fname} \
    --output-dir ${mask_output_dir} \
    --model-chkpt ${model_chkpt} \
    --model-type ${model_type} \
    --model-config ${model_config} \
    --start-idx ${start_idx} \
    --end-idx ${end_idx}
    """
}

process runUNET {
    label 'small_gpu'
    conda "${moduleDir}/envs/conda_unet.yml"

    input:
    tuple val(meta), path(image_path), val(mask_fname), val(start_idx), val(end_idx)
    val mask_output_dir
    path model_config
    path model_chkpt

    output:
    tuple val("${image_path.simpleName}"), val(mask_fname), val(mask_output_dir), val("${mask_output_dir}/${mask_fname}_${end_idx-start_idx}_${start_idx}.npy"), emit: mask

    script:
    """
    python ${moduleDir}/resources/usr/bin/run_unet.py \
    --img-path ${image_path} \
    --mask-fname ${mask_fname} \
    --output-dir ${mask_output_dir} \
    --model-chkpt ${model_chkpt} \
    --model-config ${model_config} \
    --start-idx ${start_idx} \
    --end-idx ${end_idx}
    """
}

process runMITONET {
    label 'small_gpu'
    conda "${moduleDir}/envs/conda_mitonet.yml"

    input:
    tuple val(meta), path(image_path), val(mask_fname), val(start_idx), val(end_idx)
    val mask_output_dir
    path model_config
    path model_chkpt
    val model_type

    output:
    tuple val("${image_path.simpleName}"), val(mask_fname), val(mask_output_dir), val("${mask_output_dir}/${mask_fname}_${end_idx-start_idx}_${start_idx}.npy"), emit: mask

    script:
    """
    python ${moduleDir}/resources/usr/bin/run_mitonet.py \
    --img-path ${image_path} \
    --mask-fname ${mask_fname} \
    --output-dir ${mask_output_dir} \
    --model-chkpt ${model_chkpt} \
    --model-type ${model_type} \
    --model-config ${model_config} \
    --start-idx ${start_idx} \
    --end-idx ${end_idx}
    """
}

process combineStacks {
    conda "${moduleDir}/envs/conda_combine_stacks.yml"

    input:
    tuple val(img_simplename), val(mask_fname), val(mask_output_dir), path(masks)

    output:
    stdout

    script:
    """
    python ${moduleDir}/resources/usr/bin/combine_stacks.py \
    --mask-fname ${mask_fname} \
    --output-dir ${mask_output_dir} \
    --masks ${masks}
    """
}