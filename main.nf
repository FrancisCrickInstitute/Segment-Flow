#!/usr/bin/env nextflow
nextflow.enable.dsl=2

def helpMessage() {
    log.info """\
    ==============================================
       S E G M E N T - F L O W  P I P E L I N E
    ==============================================

    Usage:
        nextflow run FrancisCrickInstitute/Segment-Flow -entry <inference|finetune> [options]

    ── Shared (required) ──────────────────────────────────────────────────────
        --model             STR     Model to use              [default: ${params.model}]
        --model_type        STR     Model variant             [default: ${params.model_type}]
        --task              STR     Task to perform           [default: ${params.task}]

    ── Shared (optional) ──────────────────────────────────────────────────────
        --help                      Show this message and exit
        --model_config      PATH    Path to model config file
        --root_dir          PATH    Root cache directory      [default: ${params.root_dir}]

    ── Inference ──────────────────────────────────────────────────────────────
      Required:
        --img_dir           PATH    CSV of image filepaths to segment

      Optional:
        --param_hash        STR     Hash of model params (auto-computed if absent)
        --preprocess        LIST    Preprocessing params (see docs)
        --postprocess       BOOL    Run postprocessing        [default: ${params.postprocess}]
        --output_format     STR     'rle' or 'tiff'           [default: ${params.output_format}]
        --output_mask_type  STR     'auto','binary','instance' [default: ${params.output_mask_type}]

    ── Finetuning ─────────────────────────────────────────────────────────────
      Required:
        --train_dir         PATH    Directory of training images
        --test_dir          PATH    Directory of test images
        --model_save_name   STR     Name for the saved model
        --model_save_dir    PATH    Directory to save the finetuned model
        --epochs            INT     Number of training epochs

      Optional:
        --finetune_layers   INT     Number of layers to finetune
        --weight_decay      FLOAT   Weight decay
        --learning_rate     FLOAT   Learning rate
        --sdg               BOOL    Use SGD optimiser
        --momentum          FLOAT   SGD momentum

    ── Profiles ───────────────────────────────────────────────────────────────
        local, crick, crick_dev, rosalind

    Docs: ${workflow.manifest.docsUrl}
    ==============================================
    """.stripIndent()
}

if ( params.help ) {
    helpMessage()
    exit 0
}

def validateParams(params) {
    def errors = []

    if ( !params.model     ) errors << "Missing required parameter: --model"
    if ( !params.model_type) errors << "Missing required parameter: --model_type"
    if ( !params.task      ) errors << "Missing required parameter: --task"

    // Type/existence checks
    if ( params.img_dir && !file(params.img_dir).exists() ) 
        errors << "img_dir does not exist: ${params.img_dir}"

    // Check output mask format is custom .rle or .tiff format
    if ( !['rle', 'tiff'].contains(params.output_format?.toLowerCase()) )
        errors << "Invalid output_format: ${params.output_format}. Must be one of 'rle' or 'tiff'."

    // Check output mask type is either binary or instance, used for outputs
    if ( !['auto', 'binary', 'instance'].contains(params.output_mask_type?.toLowerCase()) )
        errors << "Invalid output_mask_type: ${params.output_mask_type}. Must be one of 'binary', 'instance', or 'auto'."

    if ( errors ) {
        log.error "Parameter validation failed:\n" + errors.join("\n")
        exit 1
    }
}

validateParams(params)

def resolvedParamHash = params.param_hash ?: {
    // Exclude params that don't affect output content
    def excluded = ['help', 'param_hash', 'root_dir', 'output_format'] as Set
    def src = params
        .findAll { k, _v -> !(k in excluded) }
        .sort()
        .collect { k, v -> "${k}=${v}" }
        .join('|')
    java.security.MessageDigest.getInstance('MD5')
        .digest(src.bytes).encodeHex().toString()[0..7]
}()

// Default root/cache directory for masks, models etc. to be stored
def root_dir            = params.root_dir
// Construct other directories from root
def cache_dir           = "${root_dir}/aiod_cache"
def model_dir           = "${cache_dir}/${params.model}"
def model_chkpt_dir     = "${model_dir}/checkpoints"
params.model_chkpt_dir  = model_chkpt_dir  // needed by storeDir in modules

// Import processes from model modules
include { setupModel; downloadArtifact; preprocessImage; splitStacks; runModel; combineStacks; finetuneModel } from './modules/models'

def log_timestamp = new java.util.Date().format( 'yyyy-MM-dd HH:mm:ss' )

display_pipeline_info()

def display_pipeline_info() {
    log_timestamp = new java.util.Date().format( 'yyyy-MM-dd HH:mm:ss' )
    log.info """\
         ====================================================
                        AI ONDEMAND PIPELINE
                        ${log_timestamp}
         ====================================================
    """.stripIndent()
    params.each{ k, v -> log.info "${k.padRight(20)} : ${v}" }
    log.info """
         Work directory       : ${workDir}
         Profile              : ${workflow.profile}
         Full Command         : ${workflow.commandLine}
         ====================================================
    """.stripIndent()
}

// Function to get the name of the mask file given the image and model-version-task
def getMaskName(img_file, resolvedParamHash) {
    return "${img_file.baseName}" + "_masks_" + "${params.task}-${params.model}-${params.model_type}-${resolvedParamHash}"
}

// NOTE: Name this workflow when finetuning is implemented for multiple workflows
workflow inference {
    // Dynamically discover available models by scanning for run_<model>.py files
    def modelScriptsDir = file("${workflow.projectDir}/modules/models/resources/usr/bin")
    def availableModels = modelScriptsDir.listFiles()
        .findAll { it.name.startsWith('run_') && it.name.endsWith('.py') }
        .collect { it.name.replaceAll(/^run_/, '').replaceAll(/\.py$/, '') }
    assert availableModels.contains( params.model ), "Model ${params.model} not yet implemented! Available models: ${availableModels.join(', ')}"

    // Download model checkpoint if it doesn't exist
    setupModel(
        params.model,
        params.model_type,
        params.task,
        params.model_config ?: '',
    )

    // Parse each registry metadata JSON into a (name, location, type) tuple and
    // call downloadArtifact once per artifact. Each call has a single mandatory
    // output, so storeDir's cache check is always unambiguous. The optional
    // channels from setupModel act as natural gates: if a model has no config,
    // setupModel.out.model_config_meta emits nothing and downloadArtifact is
    // never scheduled for it.
    def parseMeta = { label, meta_file ->
        def m = new groovy.json.JsonSlurper().parse(meta_file)
        tuple(label, m.name, m.location, m.type, m.base_model ?: null)
    }

    // Merge all artifact metadata into one channel so downloadArtifact is only
    // called once — DSL2 does not allow reusing a process in the same workflow.
    // The label ('checkpoint', 'config', 'finetuning') is carried through as a
    // val so we can filter the mixed output channel downstream.
    downloadArtifact(
        setupModel.out.model_chkpt_meta
            | map { parseMeta('checkpoint', it) }
            | mix(
                setupModel.out.model_config_meta.map     { parseMeta('config',     it) },
                setupModel.out.model_finetuning_meta.map { parseMeta('finetuning', it) },
            )
    )

    chkpt_artifact_ch = downloadArtifact.out.artifact
        | filter { label, _file, _base_model -> label == 'checkpoint' }
        | map    { _label, file, base_model -> tuple(file, base_model) }
        | first()

    chkpt_ch = chkpt_artifact_ch | map { file, _base_model -> file }
    base_model_ch = chkpt_artifact_ch | map { _file, base_model -> base_model }
    config_ch = downloadArtifact.out.artifact
        | filter { label, _file, _base_model -> label == 'config' }
        | map    { _label, file, _base_model -> file }
        | first()

    if ( params.preprocess ) {
        // Split the CSV into individual images, so we preprocessImage distributes over each source image
        channel.fromPath(params.img_dir).splitCsv( header: true, quote: '\"' )
            | map{ row ->
                meta = row.subMap("height", "width", "num_slices", "channels")
                [
                    meta,
                    file(row.img_path),
                    getMaskName( file(row.img_path) ),
                ]
            }
            | set { img_ch1 }
        // Preprocess the images, outputting one per preprocess set
        preprocessImage( img_ch1, file(params.img_dir) )
        preprocessImage.out.prep_imgs
            | flatten()
            | map{ img -> [img.name, img] }
            | set { img_names }
        // Collect all CSVs together into original file
        preprocessImage.out.img_csv
            | collectFile(name: "all_img_info.csv", keepHeader: true)
            | set { all_img_info }
        // Split the image stacks into substacks (after model download completes)
        splitStacks( all_img_info, chkpt_ch )
    }
    // If not preprocessing, just split the stacks using the original CSV
    else {
        channel.fromPath(params.img_dir).splitCsv( header: true, quote: '\"' )
            | map{ row -> [row.img_path, file(row.img_path)]}
            | set { img_names }
        splitStacks( file(params.img_dir), chkpt_ch )
    }

    // Now prepare each substack for each (poss preprocessed) image
    // To then distribute to the model
    img_ch = splitStacks.out.csv_file.splitCsv( header: true, quote: '\"' )
        | map{ row ->
            meta = row.subMap("height", "width", "num_slices", "channels")
            [
                row.img_path,
                meta,
                getMaskName( file( row.img_path ), resolvedParamHash ),
                [
                    row.start_w.toInteger(),
                    row.end_w.toInteger(),
                    row.start_h.toInteger(),
                    row.end_h.toInteger(),
                    row.start_d.toInteger(),
                    row.end_d.toInteger()
                ]
            ]
        }
        | combine(img_names, by: 0)

    // Create the name for the mask output directory
    mask_output_dir = "${model_dir}/${params.model_type}_masks"

    // TODO: Should be delegated to a workflow in the models module?
    // Select appropriate model
    mask_out = runModel (
        img_ch,
        mask_output_dir,
        config_ch,
        chkpt_ch,
        params.model_type,
        base_model_ch,
        params.output_mask_type.toLowerCase()
    ).mask

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

    combineStacks( mask_ch, params.postprocess, params.output_format.toLowerCase(), params.output_mask_type.toLowerCase() )
}

workflow finetune {
    // Dynamically discover available models by scanning for run_<model>.py files
    def modelScriptsDir = file("${workflow.projectDir}/modules/models/resources/usr/bin")
    def availableModels = modelScriptsDir.listFiles()
        .findAll { it.name.startsWith('run_finetuning_') && it.name.endsWith('.py') }
        .collect { it.name.replaceAll(/^run_finetuning_/, '').replaceAll(/\.py$/, '') }

    assert availableModels.contains( params.model ), "Model ${params.model} not yet implemented! Available models: ${availableModels.join(', ')}"
    // Download model checkpoint if it doesn't exist
    setupModel(
        params.model,
        params.model_type,
        params.task,
        params.model_config ?: '',
    )

    // Parse each registry metadata JSON into a (name, location, type) tuple and
    // call downloadArtifact once per artifact. Each call has a single mandatory
    // output, so storeDir's cache check is always unambiguous. The optional
    // channels from setupModel act as natural gates: if a model has no config,
    // setupModel.out.model_config_meta emits nothing and downloadArtifact is
    // never scheduled for it.
    def parseMeta = { label, meta_file ->
        def m = new groovy.json.JsonSlurper().parse(meta_file)
        tuple(label, m.name, m.location, m.type, m.base_model ?: null)
    }

    // Merge all artifact metadata into one channel so downloadArtifact is only
    // called once — DSL2 does not allow reusing a process in the same workflow.
    // The label ('checkpoint', 'config', 'finetuning') is carried through as a
    // val so we can filter the mixed output channel downstream.
    downloadArtifact(
        setupModel.out.model_chkpt_meta
            | map { parseMeta('checkpoint', it) }
            | mix(
                setupModel.out.model_config_meta.map     { parseMeta('config',     it) },
                setupModel.out.model_finetuning_meta.map { parseMeta('finetuning', it) },
            )
    )

    chkpt_ch = downloadArtifact.out.artifact
        | filter { label, _file, _base_model -> label == 'checkpoint' }
        | map    { _label, file, _base_model -> file }
        | first()

    finetuneModel(
        params.model_type,
        params.model_config,
        params.epochs,
        params.finetune_layers,
        params.weight_decay,
        params.learning_rate,
        params.sdg,
        params.momentum,
        params.model_save_name,
        params.train_dir,
        params.test_dir,
        chkpt_ch,
        params.model_save_dir
        )
}

// Useful output upon completion, one way or another
workflow.onComplete {
    def end_timestamp = new java.util.Date().format( 'yyyy-MM-dd HH:mm:ss' )
    if ( workflow.success ) {
        log.info """\
                 ======================================================================
                 AIoD finished SUCCESSFULLY at ${end_timestamp} after $workflow.duration
                 ======================================================================
                 """.stripIndent()
    } else {
        log.info """\
            ======================================================================
            AIoD finished WITH ERRORS at ${end_timestamp} after $workflow.duration
            ======================================================================
            """.stripIndent()
    }
}

workflow.onError {
    log.info "ERROR: AIoD stopped with the following message: ${workflow.errorMessage}"
}
