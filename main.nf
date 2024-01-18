#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Default root directory, that gets overridden by input from Napari
params.root_dir = "${workflow.homeDir}/.nextflow/aiod/"
// Construct other directories from root
params.model_dir = "${params.root_dir}/aiod_cache/${params.model}"
params.model_chkpt_dir = "${params.model_dir}/checkpoints"
params.model_chkpt_path = "${params.model_chkpt_dir}/${params.model_chkpt_fname}"

include { downloadModel; runSAM; runUNET; runMITONET; combineStacks } from './modules/models'

def log_timestamp = new java.util.Date().format( 'yyyy-MM-dd HH:mm:ss' )

log.info """\
         AI ONDEMAND PIPELINE
         (Started at ${log_timestamp})
         =======================================
         Model name      : ${params.model}
         Model variant   : ${params.model_type}
         Model checkpoint: ${params.model_chkpt_path}
         Task            : ${params.task}
         Model config    : ${params.model_config}
         Image filepaths : ${params.img_dir}
         ---
         Cache directory : ${params.model_dir}
         Work directory  : ${workDir}
         Profile         : ${workflow.profile}
         =======================================
         """.stripIndent()

log.info "Command line arguments: $workflow.commandLine"

def getMaskName(img_file, task, model, model_type) {
    return "${img_file.simpleName}" + "_masks_" + "${params.task}-${params.model}-${params.model_type}"
}

def getIndices(meta, img_path, mask_fname, num_slices, step = 50) {
    if (num_slices < step) {
        return [meta, img_path, mask_fname, [[0, num_slices]]]
    }
    else {
        indices = []
        for (int i = 0; i < num_slices; i += step) {
            indices.add([i, i + step])
        }
        return [meta, img_path, mask_fname, indices]
    }
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
            | map{ meta, img_path, mask_fname ->
                num_slices = meta.num_slices.toInteger()
                getIndices(meta, img_path, mask_fname, num_slices)
            }
            | transpose
            | map{ meta, img_path, mask_fname, indices ->
                    [meta, img_path, mask_fname, indices[0], indices[1]]
            }

    // Create the name for the mask output directory
    mask_output_dir = "${params.model_dir}/${params.model_type}_masks"

    // TODO: This should be delegated to a workflow in the models module
    // Select appropriate model
    if( params.model == "sam" )
        mask_out = runSAM (
            img_ch,
            mask_output_dir,
            params.model_config,
            chkpt_ch,
            params.model_type
        ).mask
    else if( params.model == "unet" )
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

    mask_out
    | groupTuple
    | map{ img_name, mask_fnames, output_dirs, mask_paths ->
        [
            img_name,
            mask_fnames.first(),
            output_dirs.first(),
            mask_paths,
        ]
    }
    | set { mask_ch }

    combineStacks( mask_ch )
}

workflow.onComplete {
    def end_timestamp = new java.util.Date().format( 'yyyy-MM-dd HH:mm:ss' )
    if ( workflow.success ) {
        log.info """\
                 =======================================
                 AIoD finished SUCCESSFULLY at ${end_timestamp} after $workflow.duration $workflow.elapsedTime
                 =======================================
                 """.stripIndent()
    } else {
        log.info """\
            =======================================
            AIoD finished WITH ERRORS at ${end_timestamp} after $workflow.duration
            =======================================
            """.stripIndent()
    }
}

workflow.onError {
    log.info "ERROR: AIoD stopped with the following message: ${workflow.errorMessage}"
}