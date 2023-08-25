#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Construct params based on inputs
params.model_dir = "${launchDir}/.nextflow/cache/${params.model}"
params.model_chkpt_dir = "${params.model_dir}/checkpoints"
params.model_chkpt_path = "${params.model_chkpt_dir}/${params.model_chkpt_fname}"

include { downloadModel; runSAM; runUNET; runMITONET } from './modules/models'

log.info """\
         AI ON DEMAND PIPELINE
         ===============================
         Model Name      : ${params.model}
         Model Variant   : ${params.model_type}
         Model Checkpoint: ${params.model_chkpt_path}
         Task            : ${params.task}
         Model config    : ${params.model_config}
         Image filepaths : ${params.img_dir}
         Executor        : ${params.executor}
         """.stripIndent()

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
    // Create a tuple with the simpleName (without extension) and the full path
    img_ch = Channel.fromPath( params.img_dir )
                    .splitText()
                    .map{ [ file( it.trim() ).simpleName, it ] }
    // Create channel of file names to store the masks in
    // The first item is also the simpleName
    mask_ch = img_ch.map{ [ 
        it.first(),
        file( it.last().trim() ).simpleName + "_masks_" + "${params.model}-${params.model_type}"
    ] }
    // Join the channels
    img_mask_ch = img_ch.combine( mask_ch,  by: 0 )
                        .map { fname, fpath, mask_name -> [ fpath, mask_name ]}

    // Create the name for the mask output directory
    mask_output_dir = "${params.model_dir}/${params.model_type}_masks"

    // TODO: This should be delegated to a workflow in the models module
    // Select appropriate model
    if( params.model == "sam" )
        runSAM (
            img_mask_ch,
            mask_output_dir,
            params.model_config,
            chkpt_ch,
            params.model_type
        )
    else if( params.model == "unet" )
        runUNET (
            img_mask_ch,
            mask_output_dir,
            params.model_config,
            chkpt_ch,
        )
    else if( params.model == "mitonet" )
        runMITONET (
            img_mask_ch,
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