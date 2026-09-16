# Segment-Flow
Nextflow pipeline for running any segmentation model at scale.

## Usage
> [!NOTE]
>**For full details, see our [central documentation](https://franciscrickinstitute.github.io/aiod_docs/sections/nextflow/)**.

To get the arguments for the pipeline with a description for each, run the following command:
```
nextflow run FrancisCrickInstitute/Segment-Flow --help
```

> [!IMPORTANT]
> Pin a release tag with `-r` for any run you may need to reproduce:
> ```
> nextflow run FrancisCrickInstitute/Segment-Flow -r 0.2.0 [options]
> ```
> Without `-r`, Nextflow resolves the tip of `master`, which moves as the pipeline
> develops. Available tags are listed
> [here](https://github.com/FrancisCrickInstitute/Segment-Flow/tags). You can also point
> to specific commits or branches, as discussed in the [Nextflow docs](https://docs.seqera.io/nextflow/cli#revision-selection).
