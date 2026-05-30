# Simple-ViLMedSAM

*Bridging Vision Foundation Models with Simple Text Queries for Medical Image Segmentation*

**Accepted at CVPR 2026**

## Abstract

> Medical image segmentation is challenging due to limited annotated data, high labeling costs, and substantial image heterogeneity. Although large-scale vision foundation models (e.g., SAM) have shown great potential in > this field, existing SAM-based methods typically rely on expert-defined geometric prompts or complex clinical text prompts, which limits their generalizability across diverse medical image segmentation tasks. To overcome these challenges, we propose Simple-ViLMedSAM, a CLIP-SAM integration framework that enables high-accuracy segmentation in zero-shot and few-shot settings using only simple text queries, that is, using only basic anatomical or disease-related text labels. At its core is an Implicit Pos-Prompter (IPP), which generates attribution maps containing implicit positional cues to replace traditional geometric prompts. IPP incorporates a multi-modal information bottleneck and an affinity-based refinement strategy to ensure high-quality guidance from CLIP-SAM interactions. To further enhance segmentation, we introduce a Bidirectional Interaction Decoder (BID) that employs bidirectional cross-attention to align IPP’s positional maps with SAM's pixel-level features. By jointly modeling global semantics and local details, BID significantly improves segmentation accuracy. Extensive experiments on four public datasets demonstrate that Simple-ViLMedSAM consistently outperforms existing methods in both zero-shot and few-shot medical image segmentation tasks, using only simple text queries.

## Datasets
Public datasets used in our study:
* [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/)
* [ISIC](https://challenge.isic-archive.com/data/)
* [COVID-QU-Ex](https://www.kaggle.com/datasets/anasmohammedtahir/covidqu)
* [Chest CT](https://www.kaggle.com/datasets/polomarco/chest-ct-segmentation)

Your dataset folder under "data" should be like:
```bash
data
├──ISIC
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── ...
```
All of the masks must be binary segmentation images with 0 for background and 255 for foreground; if not, please process them accordingly.
## Guidance

### Models
Download SAM pre-trained ViT-H at [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth).
Download CLIP pre-trained ViT-B/16 at [here](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt).
Download BiomedCLIP pre-trained ViT-B/16 at [here](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224).

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
