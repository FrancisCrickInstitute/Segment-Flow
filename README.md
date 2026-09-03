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
> nextflow run FrancisCrickInstitute/Segment-Flow -r v0.1 [options]
> ```
> Without `-r`, Nextflow resolves the tip of `master`, which moves as the pipeline
> develops. Available tags are listed
> [here](https://github.com/FrancisCrickInstitute/Segment-Flow/tags).
