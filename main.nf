#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// Concatenate rather than interpolate this into another stripIndent()'d block:
// its flush-left lines would reset that block's computed indent.
def banner() {
    """\
    ════════════════════════════════════════════════════
                  █████╗ ██╗        ██████╗
                 ██╔══██╗██║        ██╔══██╗
                 ███████║██║ █████╗ ██║  ██║
                 ██╔══██║██║██╔══██╗██║  ██║
                 ██║  ██║██║╚█████╔╝██████╔╝
                 ╚═╝  ╚═╝╚═╝ ╚════╝ ╚═════╝

                   S E G M E N T - F L O W
    ════════════════════════════════════════════════════
    """.stripIndent()
}

def helpMessage() {
    log.info banner() + """\

    Usage:
        nextflow run FrancisCrickInstitute/Segment-Flow [options]

    Required:
        --img_dir       PATH    CSV of image filepaths to segment
        --model         STR     Model to use          [default: ${params.model}]
        --model_type    STR     Model variant         [default: ${params.model_type}]
        --task          STR     Task to perform       [default: ${params.task}]

    Optional:
        --help                  Show this message and exit
        --model_config  PATH    Path to model config file
        --param_hash    STR     Hash of model config
        --root_dir      PATH    Root cache directory            [default: ${params.root_dir}]
        --output_format STR     'rle' or 'tiff'                 [default: ${params.output_format}]
        --output_mask_type STR  'auto','binary','instance'      [default: ${params.output_mask_type}]
        --preprocess    LIST    Preprocessing params (see docs) [default: ${params.preprocess}]
        --postprocess   BOOL    Run postprocessing              [default: ${params.postprocess}]

    Profiles:
        local, crick, crick_dev, rosalind

    Docs: ${workflow.manifest.docsUrl}
    ════════════════════════════════════════════════════
    """.stripIndent()
}

if ( params.help ) {
    helpMessage()
    exit 0
}

def validateParams(params) {
    def errors = []

    if ( !params.img_dir   ) errors << "Missing required parameter: --img_dir"
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
include { setupModel; downloadArtifact; computeImageIds; preprocessImage; splitStacks; runModel; combineStacks } from './modules/models'

def log_timestamp = new java.util.Date().format( 'yyyy-MM-dd HH:mm:ss' )

// Could consider https://stackoverflow.com/a/71529563 for auto-printing

log.info banner() + """\
         Started         : ${log_timestamp}
         Model name      : ${params.model}
         Model variant   : ${params.model_type}
         Task            : ${params.task}
         Model config    : ${params.model_config}
         Config Hash     : ${resolvedParamHash}
         Image filepaths : ${params.img_dir}
         ---
         Cache directory : ${model_dir}
         Work directory  : ${workDir}
         Profile         : ${workflow.profile}
         ---
         Full Command    : ${workflow.commandLine}
         ════════════════════════════════════════════════════
         """.stripIndent()


// NOTE: Mirrors aiod_utils.io.get_mask_name() - keep both in sync if this changes.
// resolvedParamHash encodes all relevant params to ensure uniqueness
def getMaskName(image_id, prep_hash, resolvedParamHash) {
    def prep_suffix = prep_hash ? "_${prep_hash}" : ""
    return "${image_id}${prep_suffix}_masks_${resolvedParamHash}"
}

// TODO: Name this workflow when finetuning is implemented for multiple workflows
workflow {
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
    // call downloadArtifact once per artifact (if present)
    def parseMeta = { label, meta_file ->
        def m = new groovy.json.JsonSlurper().parse(meta_file)
        tuple(label, m.name, m.location, m.type)
    }

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
        | filter { label, _file -> label == 'checkpoint' }
        | map    { _label, file -> file }
        | first()

    config_ch = downloadArtifact.out.artifact
        | filter { label, _file -> label == 'config' }
        | map    { _label, file -> file }
        | first()

    // Registry-resolved axes for selected model-version
    model_axes_ch = setupModel.out.model_meta
        | map { new groovy.json.JsonSlurper().parse(it).axes ?: '' }
        | first()

    // Add an image_id (and placeholder prep_hash) to every row
    // image_id relies on Python-side (bioio) extension recognition that Groovy cannot replicate
    // So compute it here and carry it forward
    computeImageIds( file(params.img_dir) )
    normalized_img_dir = computeImageIds.out.csv

    if ( params.preprocess ) {
        // Split the CSV into individual images, so we preprocessImage distributes over each source image
        normalized_img_dir.splitCsv( header: true, quote: '\"' )
            | map{ row ->
                // image_id is needed by preprocessImage's own output glob
                meta = row.subMap("height", "width", "num_slices", "channels", "image_id")
                [
                    meta,
                    file(row.img_path),
                    "", // mask_fname: unused by preprocessImage's script
                ]
            }
            | set { img_ch1 }
        // Preprocess the images, outputting one per non-empty preprocess set
        // Empty sets (i.e. no-ops) are mixed in later so no copies are made
        preprocessImage( img_ch1, normalized_img_dir )
        preprocessImage.out.prep_imgs
            | flatten()
            | map{ img -> [img.name, img] }
            | set { prep_img_names }
        // Collect all CSVs together into original file
        preprocessImage.out.img_csv
            | collectFile(name: "all_img_info.csv", keepHeader: true)
            | set { all_img_info }
        // Grab the hashes and associated prep sets and log for info
        preprocessImage.out.hash_legend
            | first()
            | map { it.text }
            | subscribe { legend -> log.info "\nPreprocessing hash legend for this run:\n${legend}" }
        // Check for presence of any no-op sets & integrate if so
        if ( params.preprocess.any { it instanceof List && it.isEmpty() } ) {
            normalized_img_dir.splitCsv( header: true, quote: '\"' )
                | map{ row -> [row.img_path, file(row.img_path)] }
                | mix(prep_img_names)
                | set { img_names }
            all_img_info
                // normalized_img_dir shares all_img_info's columns
                | mix(normalized_img_dir)
                | collectFile(name: "all_img_info.csv", keepHeader: true)
                | set { all_img_info }
        } else {
            prep_img_names.set { img_names }
        }
        // Split the image stacks into substacks (after model download completes)
        splitStacks( all_img_info, chkpt_ch )
    }
    // If not preprocessing, just split the stacks using the original CSV
    else {
        normalized_img_dir.splitCsv( header: true, quote: '\"' )
            | map{ row -> [row.img_path, file(row.img_path)]}
            | set { img_names }
        splitStacks( normalized_img_dir, chkpt_ch )
    }

    // Now prepare each substack for each (poss preprocessed) image
    // To then distribute to the model
    split_csv_rows = splitStacks.out.csv_file.splitCsv( header: true, quote: '\"' )

    img_ch = split_csv_rows
        | map{ row ->
            // Reminder: resolvedParamHash is unique per run, and prep_hash unique per preprocessing branch, and image_id obvs unique per image
            meta = row.subMap("height", "width", "num_slices", "channels", "prep_hash")
            [
                row.img_path,
                meta,
                getMaskName( row.image_id, row.prep_hash, resolvedParamHash ),
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

    // Split CSV is one row per substack (i.e. per runModel job), so counting rows
    // per mask_fname gives groupTuple the group size it needs to emit early.
    // Keyed on mask_fname to match runModel's output key exactly
    substack_counts = split_csv_rows
        | map { row -> tuple( getMaskName( row.image_id, row.prep_hash, resolvedParamHash ), 1 ) }
        | groupTuple()
        | map { mask_fname, ones -> tuple( mask_fname, ones.size() ) }

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
        params.output_mask_type.toLowerCase(),
        model_axes_ch
    ).mask

    // Group all the substacks per image+branch together to combine.
    // mask_fname (image_id-based) is the canonical identity to group on
    // (as it is image+preprocessing-specific)
    // groupKey(mask_fname, n_substacks) tells groupTuple num substacks for each image
    // emitting those dataset masks when those are done
    mask_out
    | combine( substack_counts, by: 0 )
    | map{ mask_fname, meta, output_dir, mask_path, n_substacks ->
        tuple( groupKey(mask_fname, n_substacks), meta, output_dir, mask_path )
    }
    | groupTuple()
    | map{ mask_fname, meta, output_dirs, mask_paths ->
        [
            mask_fname,
            meta.first(),
            params.model,
            output_dirs.first(),
            mask_paths,
        ]
    }
    | set { mask_ch }

    combineStacks( mask_ch, params.postprocess, params.output_format.toLowerCase(), params.output_mask_type.toLowerCase() )
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
