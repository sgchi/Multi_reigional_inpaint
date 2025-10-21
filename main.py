import os
import torch
import argparse

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from utils import data_preprocess
from multi_regional_inpaint import MultiRegionalInpaintPipeline
from transformers import SamModel, SamProcessor


class Convex_Amodal(object):
    def __init__(self, eval=False):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = "runwayml/stable-diffusion-inpainting"

        self.model = MultiRegionalInpaintPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
            use_safetensors=False
        ).to(self.device)

        if eval:
            from torchmetrics.multimodal.clip_score import CLIPScore
            self.metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
            self.score_list = []
            self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(self.device)
            self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
            self.IoU_list = []
            self.Mp_list = []
            self.Ms_list = []

    def get_clip_score(self, img, txt):
        self.score_list.append(self.metric(torch.from_numpy(img.copy()), txt).detach().numpy())

    def evaluate(self, batch, exp_type="convex_hull_mask"):
        human_mask, gt_mask, obj_mask, contact_mask, convex_mask, prompt = batch

        prompt = batch['obj_name']
        for idx, (completed_img, gt_mask, occludee_mask, hum_mask, conv_mask, txt) in enumerate(zip(completed_img, gt_mask, obj_mask, human_mask, convex_mask, prompt)):
            points = (occludee_mask.permute(0,2,1).bool() & ~hum_mask.permute(0,2,1).bool()).squeeze().nonzero()
            rand_idx = np.random.choice(len(points), 10)
            inputs = self.sam_processor(completed_img, input_points=[[points[rand_idx].float().tolist()]], return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.sam_model(**inputs)

            sam_i = 2
            sam_masks = self.sam_processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )
            self.score_list.append(self.metric(torch.from_numpy(completed_img.copy())*sam_masks[0][0,sam_i].unsqueeze(-1), txt).detach().numpy())

            union = (np.array(gt_mask.squeeze().bool()) | np.array(sam_masks[0][0,sam_i].bool().squeeze())).sum()
            inter = (np.array(gt_mask.squeeze().bool()) & np.array(sam_masks[0][0,sam_i].bool().squeeze())).sum()
            self.IoU_list.append((inter/union*100).item())
            Mp_ratio = (np.array(gt_mask.squeeze().bool()) & np.array(convex_mask.bool().squeeze())).sum() / np.array(convex_mask.bool().squeeze()).sum()
            Ms_ratio = (np.array(gt_mask.squeeze().bool()) & np.array(hum_mask.bool().squeeze()) & ~np.array(convex_mask.bool().squeeze())).sum() / (np.array(hum_mask.bool().squeeze()) & ~np.array(convex_mask.bool().squeeze())).sum()
            self.Mp_list.append((Mp_ratio*100).item())
            self.Ms_list.append((Ms_ratio*100).item())

    def object_amodal_completion(self, batch, T_hat=[0.5], save=True):
        guidance_scale= 7.5
        num_samples = 1
        generator = torch.Generator(device=self.device).manual_seed(0) # change the seed to get different results

        M_p, occluder_mask, occludee_mask, prompt =\
            batch["M_p"], batch["occluder_mask"], batch["occludee_mask"], batch['obj_name']

        regions = [M_p.to(self.device).squeeze(), torch.max(M_p, occluder_mask).to(self.device).squeeze()] #occluder_mask == M_p U M_s
        orig_img = (torch.stack((batch['images'],)) * occludee_mask).to(self.device)

        output = []
        for r in T_hat:
            images = self.model(
                prompt= prompt,
                image=orig_img,
                mask_image=regions,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=num_samples,
                num_inference_steps=50,
                strength=[1.0, r]
            ).images

            output.append(images[0])

        if save:
            assert len(T_hat) == 4
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))

            def process_tensor(tensor):
                img_np = tensor.detach().cpu().numpy()
                if img_np.ndim == 4:
                    img_np = img_np[0]
                if img_np.shape[0] in [1, 3, 4]: # Check for channel-first format
                    img_np = np.transpose(img_np, (1, 2, 0))
                return np.squeeze(img_np)

            axes[0, 0].imshow(process_tensor(batch['images']))
            axes[0, 0].set_title("Input Image (Batch)")

            axes[0, 1].imshow(process_tensor(orig_img))
            axes[0, 1].set_title("Masked Original Image")

            axes[0, 2].imshow(process_tensor(M_p), cmap='gray')
            axes[0, 2].set_title("$M_p$")

            axes[0, 3].imshow(process_tensor(occluder_mask), cmap='gray')
            axes[0, 3].set_title("$M_p \ U \  M_s$")

            for i, img in enumerate(output):
                axes[1, i].imshow(img)
                axes[1, i].set_title(f"Strength $r = {T_hat[i]}$")

            for ax in axes.flat:
                ax.axis('off')

            plt.tight_layout()
            plt.show()
            if not os.path.exists("results"):
                os.mkdir("results/")
            plt.savefig("results/"+ prompt + ".jpg")
        return images

def main(demo_data_path):
    input_bundle = data_preprocess(Path(demo_data_path))
    pipe = Convex_Amodal()
    pipe.object_amodal_completion(input_bundle, T_hat=[0.1, 0.5, 0.8, 0.9])

if __name__ == '__main__':
# 1. Create the parser
    parser = argparse.ArgumentParser(description="Amodal completion script.")
    parser.add_argument("--input_path", type=str, help="Path to the demo data directory or file.")
    args = parser.parse_args()
    main(args.input_path)
