# Segment-Flow

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Nextflow](https://img.shields.io/badge/nextflow-DSL2-23aa62.svg)](https://www.nextflow.io/)
[![Docs](https://img.shields.io/badge/docs-aiod__docs-1f6feb.svg)](https://franciscrickinstitute.github.io/aiod_docs/sections/nextflow/)

Nextflow pipeline for running any segmentation model at scale.

Segment-Flow is the computation core of [AI OnDemand (AIoD)](https://franciscrickinstitute.github.io/aiod_docs): it splits images into substacks, builds an isolated conda environment per model family, fetches checkpoints, runs the model in parallel, and combines results at the end. It can be driven headlessly from the terminal, or through a front-end such as the [Napari plugin](https://github.com/FrancisCrickInstitute/aiod_napari).

## Requirements

- [Nextflow](https://www.nextflow.io/docs/latest/install.html) (which needs Java 17+)
- Conda (per-model-family environments are built automatically)

See the docs [Prerequisites](https://franciscrickinstitute.github.io/aiod_docs/sections/getting_started/#prerequisites) for installing these, including on HPC.

There is nothing to install for the pipeline itself — Nextflow pulls it from GitHub on first run.

## Usage

> [!NOTE]
> **For full details, see our [central documentation](https://franciscrickinstitute.github.io/aiod_docs/sections/nextflow/)**.

A minimal run needs four parameters — an input CSV describing your images, plus the model family, model version and task:

```bash
nextflow run FrancisCrickInstitute/Segment-Flow -r 0.2.1 \
    -profile local \
    --img_dir /path/to/all_img_paths.csv \
    --model empanada \
    --model_type mitonet_v1 \
    --task mito
```

Where `all_img_paths.csv` has one row per image:

```csv
img_path,num_slices,height,width,channels
/path/to/image.tiff,50,1024,1024,1
```

Valid values for `--model`, `--model_type` and `--task` come from the [model registry](https://github.com/FrancisCrickInstitute/aiod_registry) and are listed in the docs under [Available Models](https://franciscrickinstitute.github.io/aiod_docs/sections/model_registry/models/). Everything else has a default.

To get the arguments for the pipeline with a description for each, run the following command:

```
nextflow run FrancisCrickInstitute/Segment-Flow --help
```

> [!IMPORTANT]
> Pin a release tag with `-r` for any run you may need to reproduce:
> ```
> nextflow run FrancisCrickInstitute/Segment-Flow -r 0.2.1 [options]
> ```
> Without `-r`, Nextflow resolves the tip of `master`, which moves as the pipeline
> develops. Available tags are listed
> [here](https://github.com/FrancisCrickInstitute/Segment-Flow/tags). You can also point
> to specific commits or branches, as discussed in the [Nextflow docs](https://docs.seqera.io/nextflow/cli#revision-selection).

## Documentation

Full documentation for AIoD lives at **[franciscrickinstitute.github.io/aiod_docs](https://franciscrickinstitute.github.io/aiod_docs/)**.

| Topic | Link |
| --- | --- |
| Never run it from the terminal before? | [Your First Headless Run](https://franciscrickinstitute.github.io/aiod_docs/sections/getting_started/first_headless_run/) |
| Pipeline steps and full parameter reference | [Nextflow Pipeline](https://franciscrickinstitute.github.io/aiod_docs/sections/nextflow/) |
| Writing the input CSV | [Creating the input CSV](https://franciscrickinstitute.github.io/aiod_docs/sections/nextflow/#creating-the-input-csv) |
| Substacks, overlap and throughput | [Tuning the pipeline](https://franciscrickinstitute.github.io/aiod_docs/sections/nextflow/#tuning-the-pipeline) |
| Where results are cached, and reproducibility | [AIoD Concepts](https://franciscrickinstitute.github.io/aiod_docs/sections/concepts/#caching) |
| Adding a model family to the pipeline | [Expanding AIoD](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/expanding/#add-a-new-model-family_1) |
| Adding an execution profile for your site | [Add a profile](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/expanding/#add-a-profile) |
| Something went wrong | [Troubleshooting](https://franciscrickinstitute.github.io/aiod_docs/sections/support/troubleshooting/) |

## Contributing

Contributions are very welcome! See [Expanding AIoD](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/expanding/) to add a model family or an execution profile, and the [AIoD Developer Guide](https://franciscrickinstitute.github.io/aiod_docs/sections/contributing/developing/) for setting up across the AIoD repos.

Adding a new model family needs two things: a manifest entry in [aiod_registry](https://github.com/FrancisCrickInstitute/aiod_registry), and a matching `run_<short_name>.py` plus conda environment here, keyed by the manifest's `short_name`.

Pipeline tests live in [`tests/`](tests) and are run with [nf-test](https://www.nf-test.com/):

```bash
nf-test test
```

## Support

Please [open an issue](https://github.com/FrancisCrickInstitute/Segment-Flow/issues) for bugs or feature requests, including the Nextflow log (`.nextflow.log`) and the profile you ran with. For usage problems, start with [Troubleshooting](https://franciscrickinstitute.github.io/aiod_docs/sections/support/troubleshooting/).

## License

MIT — see [LICENSE](LICENSE).
