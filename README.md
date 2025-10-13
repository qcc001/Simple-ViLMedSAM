# Simple-ViLMedSAM
## Abstract

Medical image segmentation is challenging due to limited annotated data, high labeling costs, and substantial image heterogeneity. Although large-scale vision foundation models (e.g., SAM) have shown great potential in this field, existing SAM-based methods typically rely on expert-defined geometric prompts or complex clinical text prompts, which limits their generalizability across diverse medical image segmentation tasks. To overcome these challenges, we propose Simple-ViLMedSAM, a CLIP-SAM integration framework that enables high-accuracy segmentation in zero-shot and few-shot settings using only simple text queries, that is, using only basic anatomical or disease-related text labels. At its core is an Implicit Pos-Prompter (IPP), which generates attribution maps containing implicit positional cues to replace traditional geometric prompts. IPP incorporates a multi-modal information bottleneck and an affinity-based refinement strategy to ensure high-quality guidance from CLIP-SAM interactions. To further enhance segmentation, we introduce a Bidirectional Interaction Decoder (BID) that employs bidirectional cross-attention to align IPP’s positional maps with SAM's pixel-level features. By jointly modeling global semantics and local details, BID significantly improves segmentation accuracy. Extensive experiments on four public datasets demonstrate that Simple-ViLMedSAM consistently outperforms existing methods in both zero-shot and few-shot medical image segmentation tasks, using only simple text queries.

## Datasets
Public datasets used in our study:
* [COVID-QU-Ex](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu)

## Guidance

### Installation

```bash
pip install -r requirements.txt
```

### Training
```bash
python train.py
```

### Testing
```bash
python test.py
```
