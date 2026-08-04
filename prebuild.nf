#!/usr/bin/env nextflow
nextflow.enable.dsl=2

// ---------------------------------------------------------------------------
// Prebuild workflow
// ---------------------------------------------------------------------------
// Triggers conda environment creation for every conda spec used by the
// pipeline, without requiring real input data.  Each process body is a no-op
// ("echo done") — the only goal is for Nextflow to resolve and build the
// conda environment into the shared cacheDir before any user jobs run.
//
// Usage (single model):
//   nextflow run prebuild.nf -profile crick --model cellpose
//
// Usage (all models — preferred):
//   ./prebuild_all.sh crick
// ---------------------------------------------------------------------------

// Fixed env: preprocessImage / splitStacks / combineStacks
process prebuildCombineStacks {
    conda "${moduleDir}/modules/models/envs/conda_combine_stacks.yml"

    script:
    "echo 'conda env ready: combine_stacks'"
}

// Fixed env: setupModel / downloadArtifact
process prebuildSetupModel {
    conda "${moduleDir}/modules/models/envs/conda_setup_model.yml"

    script:
    "echo 'conda env ready: setup_model'"
}

// Per-model env: runModel
// conda path mirrors the runModel process exactly; task.ext.condaDir is
// injected by the active profile (e.g. "cuda" on crick, "generic" on local).
process prebuildRunModel {
    conda "${moduleDir}/modules/models/envs/${task.ext.condaDir}/conda_${params.model}.yml"

    script:
    "echo 'conda env ready: ${params.model} (${task.ext.condaDir})'"
}

workflow {
    prebuildCombineStacks()
    prebuildSetupModel()
    prebuildRunModel()
}
