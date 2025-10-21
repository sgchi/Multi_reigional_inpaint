# Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting

### PyTorch Implementation

This repository contains the code for the [**paper**](https://arxiv.org/abs/2508.00427) on **amodal completion** for human-object interactions. Our model inpaints the occluded parts during HOI, creating a complete picture of the scene.

---

## ⚙️ Installation

Getting the environment set up is straightforward. We recommend using a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/sgchi/multi-regional-inpainting.git
    cd multi-regional-inpainting
    ```

2.  **Install dependencies:**
    This project is built using Python 3.8+ and PyTorch. The main dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    *Key dependencies include:*
    * `torch==2.4.1`
    * `diffusers==0.31.0`
    * `transformers==4.46.1`
---

## 🚀 Demo

You can run a quick demo on sample images to see the model in action.

```bash
python main.py --input_path data/behave/sequences/Date03_Sub04_monitor_move/t0003.000/
```
The script will process images in the data/behave/ folder and save the inpainted results to results/.


## 📝 Roadmap & Future Work

We are committed to the ongoing development of this project.

[x] Initial demo code release

[ ] Support for inference on in-the-wild images

[ ] Release code for the novel pose synthesis application


## 📜 Citation

If you use this code or our work in your research, please cite our paper:
```
@article{chi2025contact,
  title={Contact-Aware Amodal Completion for Human-Object Interaction via Multi-Regional Inpainting},
  author={Chi, Seunggeun and Sachdeva, Enna and Huang, Pin-Hao and Lee, Kwonjoon},
  journal={arXiv preprint arXiv:2508.00427},
  year={2025}
}
```

## Acknowledgement

Our implementation builds upon code from [diffusers](https://github.com/huggingface/diffusers) and [HDM](https://github.com/xiexh20/HDM). We thank the original authors for their contributions.
