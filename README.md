<div align="center">
  <h1>Hyperbolic Safety-Aware Vision-Language Models

  (CVPR 2025)
  </h1>  
</div>

This repository contains the reference code for the paper [**Hyperbolic Safety-Aware Vision-Language Models**](https://arxiv.org/abs/2503.12127).

***Warning:** This project involves explicit sexual content, racially insensitive language, and other material that may be harmful or disturbing to certain users. Please use this content solely for research purposes and proceed with caution.*

<p align="center">
  <img src="imgs/hysac-method.png" alt="HySAC" width="820" />
</p> 

## Table of Contents

1. [Overview](#overview)
2. [Usage of Hysac](#usage-of-hysac)
3. [Citation](#citation)

## Overview
HySAC, Hyperbolic Safety-Aware CLIP, models hierarchical safety relations to enable effective retrieval of unsafe content, dynamically redirecting it to **safer** alternatives for enhanced content moderation.

**Useful Links**
- [🤗 HuggingFace HySAC Model](https://huggingface.co/aimagelab/HySAC)
- [📄 Paper](https://arxiv.org/abs/2503.12127)
- [🎯 Project Page](https://aimagelab.github.io/HySAC/)

## Installation
```
git clone https://github.com/aimagelab/HySAC.git
cd HySAC
conda create -n safe-clip python==3.10
conda activate safe-clip
pip install -r requirements.txt
pip install -e .
```

## Usage of HySAC
See the snippet to use **HySAC**:

```python
from hysac.models import HySAC

model_id = "aimagelab/hysac"
model = HySAC.from_pretrained(model_id, device="cuda").to("cuda")
```

Use standard methods `encode_image` and `encode_text` to encode images and text with the model.
Before using the model to retrieve data, you can apply the safety traversal to the query embedding by calling the `traverse_to_safe_image` and `traverse_to_safe_text` methods.

# Citation
Please cite with the following BibTeX:
```
@inproceedings{poppi2025hyperbolic,
  title={{Hyperbolic Safety-Aware Vision-Language Models}},
  author={Poppi, Tobia and Kasarla, Tejaswi and Mettes, Pascal and Baraldi, Lorenzo and Cucchiara, Rita},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
