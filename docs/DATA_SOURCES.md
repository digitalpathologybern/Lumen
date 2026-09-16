# Whole-slide cohort sources

`configs/inference.yaml` names two slide-level tables under `datasets/`. **Neither
table ships**, and no whole-slide image, annotation or label ships either:
`datasets/` is gitignored, and each table locates its slides by an absolute
`file_path` you repoint at your own copy.

This file is the whole-slide analogue of [`MODEL_SOURCES.md`](MODEL_SOURCES.md):
it is how the external cohorts are obtained, so the external half of the
slide-level evaluation can be rebuilt from public data plus this repository.

## Internal table

Lymph-node slides from the Institute of Tissue Medicine and Pathology,
University of Bern (`breast`, `crc`, `endometrial`, `lung`, `upper_gi`).
Patient data held under ethics approval. **Not distributed, and not
reconstructible from public sources.** See the paper's Data availability
statement for the request route. Every stage that needs this table is marked as
such in [`../REPRODUCE.md`](../REPRODUCE.md).

## External table

One row per slide, with columns `dataset, case, file_name, file_path, label,
notes`. `dataset` is the cohort key used everywhere else in the code (it becomes
the `group` column, and the `<group>/` directory under the embedding cache);
`file_path` is a local absolute path and has to be repointed at wherever the
downloaded slides land.

`rows` counts the table; `evaluable` is what `load_slide_table` keeps after
label mapping, dropping `discard` / `itc` / `notLN` / `ask pathologist` and the
rest of the unmappable labels.

| `dataset` | Source | Where to download | rows | evaluable (pos/neg) |
| --- | --- | --- | --- | --- |
| `breast_camelyon17` | CAMELYON17. Litjens et al., *GigaScience* 7(6), 2018, [doi:10.1093/gigascience/giy065](https://doi.org/10.1093/gigascience/giy065) | <http://gigadb.org/dataset/100439>, mirror of <https://camelyon17.grand-challenge.org/Data/> | 964 | 921 (306/615) |
| `breast_histai` | HISTAI. Nechaev et al., [arXiv:2505.12120](https://arxiv.org/abs/2505.12120) | <https://huggingface.co/datasets/histai/HISTAI-breast> | 143 | 111 (34/77) |
| `crc_histai_b1` | HISTAI. Nechaev et al., [arXiv:2505.12120](https://arxiv.org/abs/2505.12120) | <https://huggingface.co/datasets/histai/HISTAI-colorectal-b1> | 288 | 256 (51/205) |
| `endometrial_histai` | HISTAI. Nechaev et al., [arXiv:2505.12120](https://arxiv.org/abs/2505.12120) | <https://huggingface.co/datasets/histai/HISTAI-mixed>, endometrial cases selected by hand; there is no standalone endometrial repo | 90 | 90 (13/77) |
| `lung_histai_thorax` | HISTAI. Nechaev et al., [arXiv:2505.12120](https://arxiv.org/abs/2505.12120) | <https://huggingface.co/datasets/histai/HISTAI-thorax> | 107 | 81 (42/39) |
| `melanoma_histai_b1` | HISTAI. Nechaev et al., [arXiv:2505.12120](https://arxiv.org/abs/2505.12120) | <https://huggingface.co/datasets/histai/HISTAI-skin-b1> | 306 | 98 (38/60) |
| `melanoma_histai_b2` | HISTAI. Nechaev et al., [arXiv:2505.12120](https://arxiv.org/abs/2505.12120) | <https://huggingface.co/datasets/histai/HISTAI-skin-b2> | 266 | 164 (71/93) |
| `gastric` | Gastric cancer lymph node data set. Chen, Wang, Liu & Yu, 2020, released with Wang et al., *Nat Commun* 12:1637, [doi:10.1038/s41467-021-21674-7](https://doi.org/10.1038/s41467-021-21674-7) | <https://figshare.com/articles/dataset/Gastric_cancer_lymph_node_data_set/13065986>. 500 `20201007_*.tif`, one per table row | 500 | 477 (225/252) |
| `lung_yuritolkach` | Institutional cohort, University Hospital Cologne (UKK) | **Not public.** See the paper's Data availability statement | 255 | 255 (136/119) |

All six HISTAI rows are subsets of one dataset and share one source:
Nechaev, D., Pchelnikov, A. & Ivanova, E., *HISTAI: An Open-Source, Large-Scale
Whole Slide Image Dataset for Computational Pathology*, arXiv:2505.12120 (2025).
The `dataset` key names which subset; the Hugging Face repository in the
download column is the subset itself. The slides are released under **CC BY-NC
4.0**, i.e. non-commercial. That is the licence on the data. The preprint
itself is CC BY 4.0, which does not extend to the slides.

These nine are the study's external evaluation. The table may also carry rows
for cohorts outside that scope, so its row count will not match the nine above.
`exclude_benchmark_groups` in `lumen/wsi/cohorts.py` drops them, and every stage
calls it, so all three see the same cohort set.

### Evaluable is not the same as scored

`evaluable` counts label mapping only. Extraction then applies the tissue filter
in `configs/inference.yaml`: a slide with no passing tile yields no embedding
and is absent from scoring. This is deterministic, so the same thresholds drop
the same slides on a re-run.

Two cohorts differ visibly between the two counts:

| `dataset` | evaluable | scored |
| --- | --- | --- |
| `lung_histai_thorax` | 81 | 74 |
| `gastric` | 477 | 399 |

Gastric carries most of it, 78 slides, consistent with its tile counts: a median
of 52 per slide where every other cohort sits between 196 and 786. The
manuscript's cohort table reports the scored counts.

### Labels

The label column is the study's own slide-level ground truth and is not
redistributed with the images. CAMELYON17 is the exception: its `label` values
(`negative` / `macro` / `micro` / `itc`) are the challenge's published
slide-level labels, and `load_slide_table` maps `macro` and `micro` to positive
and drops `itc`.

`gastric` carries no case identifier, so `normalize_external_case` falls back to
one patient cluster per slide; every bootstrap over that cohort therefore
resamples slides, not patients.
