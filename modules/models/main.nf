
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
    tuple path(image_path), val(mask_fname)
    val mask_output_dir
    path model_config
    path model_chkpt
    val model_type

    output:
    // Because we are manually saving it in the .cache so napari can watch for each slice
    stdout

    script:
    """
    python ${moduleDir}/resources/usr/bin/run_sam.py \
    --img-path ${image_path} \
    --mask-fname ${mask_fname} \
    --output-dir ${mask_output_dir} \
    --model-chkpt ${model_chkpt} \
    --model-type ${model_type} \
    --model-config ${model_config}
    """
}

process runUNET {
    label 'small_gpu'
    conda "${moduleDir}/envs/conda_unet.yml"

    input:
    tuple path(image_path), val(mask_fname)
    val mask_output_dir
    path model_config
    path model_chkpt

    output:
    stdout

    script:
    """
    which python
    python ${moduleDir}/resources/usr/bin/run_unet.py \
    --img-path ${image_path} \
    --mask-fname ${mask_fname} \
    --output-dir ${mask_output_dir} \
    --model-chkpt ${model_chkpt} \
    --model-config ${model_config}
    """
}

process runMITONET {
    label 'small_gpu'
    conda "${moduleDir}/envs/conda_mitonet.yml"

    input:
    tuple path(image_path), val(mask_fname)
    val mask_output_dir
    path model_config
    path model_chkpt
    val model_type

    output:
    stdout

    script:
    """
    python ${moduleDir}/resources/usr/bin/run_mitonet.py \
    --img-path ${image_path} \
    --mask-fname ${mask_fname} \
    --output-dir ${mask_output_dir} \
    --model-chkpt ${model_chkpt} \
    --model-type ${model_type} \
    --model-config ${model_config}
    """
}