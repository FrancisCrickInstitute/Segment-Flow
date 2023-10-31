#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Construct params based on inputs
params.model_dir = "${workflow.homeDir}/.nextflow/aiod/cache/${params.model}"
params.model_chkpt_dir = "${params.model_dir}/checkpoints"
params.model_chkpt_path = "${params.model_chkpt_dir}/${params.model_chkpt_fname}"

include { downloadModel; runSAM; runUNET; runMITONET } from './modules/models'

log.info """\
         AI ON DEMAND PIPELINE
         ===============================
         Model name      : ${params.model}
         Model variant   : ${params.model_type}
         Model checkpoint: ${params.model_chkpt_path}
         Task            : ${params.task}
         Model config    : ${params.model_config}
         Image filepaths : ${params.img_dir}
         ---
         Cache directory : ${params.model_dir}
         Work directory  : ${workDir}
         """.stripIndent()

def getMaskName(img_file, task, model, model_type) {
    return "${img_file.simpleName}" + "_masks_" + "${params.task}-${params.model}-${params.model_type}"
}

workflow {
    // TODO: Move the model-based stuff into a workflow under the models module
    // Download model checkpoint if it doesn't exist
    chkpt_file = file( params.model_chkpt_path )

    if ( !chkpt_file.exists() ) {
        downloadModel (
            params.model_chkpt_path,
            params.model_chkpt_loc,
            params.model_chkpt_type,
            params.model_chkpt_fname
        )
        chkpt_ch = downloadModel.out.model_chkpt
    }
    else {
        chkpt_ch = chkpt_file
    }

    // Create channel from paths to each image file
    // with the metadata and mask name too
    img_ch = Channel.fromPath( params.img_dir )
            | splitCsv( header: true )
            | map{ row ->
                meta = row.subMap("num_slices", "height", "width")
                [
                    meta,
                    row.img_path,
                    getMaskName( file(row.img_path), params.task, params.model, params.model_type )
                ]
            }

    // Create the name for the mask output directory
    mask_output_dir = "${params.model_dir}/${params.model_type}_masks"

    // TODO: This should be delegated to a workflow in the models module
    // Select appropriate model
    if( params.model == "sam" )
        runSAM (
            img_ch,
            mask_output_dir,
            params.model_config,
            chkpt_ch,
            params.model_type
        )
    else if( params.model == "unet" )
        runUNET (
            img_ch,
            mask_output_dir,
            params.model_config,
            chkpt_ch,
        )
    else if( params.model == "mitonet" )
        runMITONET (
            img_ch,
            mask_output_dir,
            params.model_config,
            chkpt_ch,
            params.model_type
        )
    else
        error "Model ${params.model} not yet implemented!"
}

workflow.onComplete{
    log.info ( workflow.success ? '\nDone!' : '\nSomething went wrong!' )
}