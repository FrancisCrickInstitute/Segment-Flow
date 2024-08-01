#!/usr/bin/env nextflow
nextflow.enable.dsl=2

def helpMessage() {
    log.info """
    Usage:

    nextflow run FrancisCrickInstitute/Segment-Flow <ARGUMENTS>

    Required arguments:
    --profile <PROFILE>          Nextflow profile to use
    --root_dir <ROOT_DIR>        Root directory for the pipeline (for caching models, masks etc.)
    --img_dir <IMG_DIR>          Directory containing images to segment
    --model <MODEL>              Model to use for segmentation
    --model_type <MODEL_TYPE>    Type of model to use for segmentation
    --task <TASK>                Task to perform with the model
    --model_config <CONFIG>      Path to model config file
    --model_chkpt_loc <LOC>      Location of model checkpoint
    --model_chkpt_type <TYPE>    Type of model checkpoint (file/url)
    --model_chkpt_fname <FNAME>  Filename of model checkpoint
    --param_hash <HASH>          Hash of the model config file. Usually generated by the front-end.
    
    Optional arguments:
    --help                      Print this help message
    """.stripIndent()
}

if ( params.help ) {
    helpMessage()
    exit 0
}

// Default root directory, that gets overridden by input from Napari
params.root_dir = "${workflow.homeDir}/.nextflow/aiod"
params.cache_dir = "${params.root_dir}/aiod_cache"
// Construct other directories from root
params.model_dir = "${params.cache_dir}/${params.model}"
params.model_chkpt_dir = "${params.model_dir}/checkpoints"
params.model_chkpt_path = "${params.model_chkpt_dir}/${params.model_chkpt_fname}"

// Import processes from model modules
include { downloadModel; splitStacks; runSAM; runSAM2; runUNET; runMITONET; combineStacks } from './modules/models'

def log_timestamp = new java.util.Date().format( 'yyyy-MM-dd HH:mm:ss' )

log.info """\
         ====================================================
                        AI ONDEMAND PIPELINE
                        ${log_timestamp}
         ====================================================
         Model name      : ${params.model}
         Model variant   : ${params.model_type}
         Model checkpoint: ${params.model_chkpt_path}
         Task            : ${params.task}
         Model config    : ${params.model_config}
         Config Hash     : ${params.param_hash}
         Image filepaths : ${params.img_dir}
         ---
         Cache directory : ${params.model_dir}
         Work directory  : ${workDir}
         Profile         : ${workflow.profile}
         ====================================================
         """.stripIndent()

// Print the command line arguments used for traceability
log.info "Command line arguments: $workflow.commandLine"

// Function to get the name of the mask file given the image and model-version-task
def getMaskName(img_file) {
    return "${img_file.simpleName}" + "_masks_" + "${params.task}-${params.model}-${params.model_type}-${params.param_hash}"
}

// NOTE: Name this workflow when finetuning is implemented for multiple workflows
workflow {
    // TODO: Move the model-based stuff into a workflow under the models module?

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

    // Split the image stacks into substacks
    // Python handles the file saving and overwrites params.img_dir
    splitStacks( params.img_dir )
    img_ch = splitStacks.out.csv_file.splitCsv( header: true, quote: '\"' )
        | map{ row ->
            meta = row.subMap("height", "width", "num_slices", "channels")
            [
                meta,
                row.img_path,
                getMaskName( file(row.img_path) ),
                [
                    row.start_h.toInteger(),
                    row.end_h.toInteger(),
                    row.start_w.toInteger(),
                    row.end_w.toInteger(),
                    row.start_d.toInteger(),
                    row.end_d.toInteger()
                ]
            ]
        }

    // Create the name for the mask output directory
    mask_output_dir = "${params.model_dir}/${params.model_type}_masks"

    // TODO: Should be delegated to a workflow in the models module?
    // Select appropriate model
    if( params.model == "sam" )
        mask_out = runSAM (
            img_ch,
            mask_output_dir,
            params.model_config,
            chkpt_ch,
            params.model_type
        ).mask
    else if( params.model == "sam2" )
        mask_out = runSAM2 (
            img_ch,
            mask_output_dir,
            params.model_config,
            chkpt_ch,
            params.model_type
        ).mask
    else if( params.model == "seai_unet" )
        mask_out = runUNET (
            img_ch,
            mask_output_dir,
            params.model_config,
            chkpt_ch,
        ).mask
    else if( params.model == "mitonet" )
        mask_out = runMITONET (
            img_ch,
            mask_output_dir,
            params.model_config,
            chkpt_ch,
            params.model_type
        ).mask
    else
        error "Model ${params.model} not yet implemented!"

    // Group all the outputs per image together to combine
    mask_out
    | groupTuple
    | map{ img_name, meta, mask_fnames, output_dirs, mask_paths ->
        [
            img_name,
            meta.first(),
            params.model,
            mask_fnames.first(),
            output_dirs.first(),
            mask_paths,
        ]
    }
    | set { mask_ch }

    combineStacks( mask_ch, params.postprocess )
}

// Useful output upon completion, one way or another
workflow.onComplete {
    def end_timestamp = new java.util.Date().format( 'yyyy-MM-dd HH:mm:ss' )
    if ( workflow.success ) {
        log.info """\
                 ================================================
                 AIoD finished SUCCESSFULLY at ${end_timestamp} after $workflow.duration
                 ================================================
                 """.stripIndent()
    } else {
        log.info """\
            ================================================
            AIoD finished WITH ERRORS at ${end_timestamp} after $workflow.duration
            ================================================
            """.stripIndent()
    }
}

workflow.onError {
    log.info "ERROR: AIoD stopped with the following message: ${workflow.errorMessage}"
}