import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from iglovikov_helper_functions.utils.image_utils import load_rgb
from check_orientation.pre_trained_models import create_model

def get_parse():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-i", "--input_path", type=Path, help="Path with images.", required=True)
    arg("-o", "--output_path", type=Path, help="Path to save masks.", required=True)
    return parser.parse_args()

def main():
    args = get_parse()

    # Create output directory if it doesn't exist
    args.output_path.mkdir(parents=True, exist_ok=True)

    model = create_model("swsl_resnext50_32x4d")
    model.eval()

    ori_90 = []
    ori_180 = []
    ori_270 = []

    for img_path in args.input_path.glob("*.png"):
        image = load_rgb(img_path)

        # Convert the NumPy array to a PyTorch tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            prediction = model(image_tensor).numpy()

        # Construct output path similar to input path but within the result folder
        relative_path = img_path.relative_to(args.input_path)
        output_path = args.output_path / relative_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
        output_filename = output_path / f"{img_path.stem}.png"

        # Plot and save the resulting images
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(image)
        
        orientation = prediction[0]
        max_position = np.argmax(orientation)

        if (max_position == 1): 
          ori_90.append(img_path)
          ax.set_title(" ".join("90"))
        elif (max_position == 2): 
           ori_180.append(img_path)
           ax.set_title(" ".join("180"))
        elif (max_position == 3):
           ori_270.append(img_path)
           ax.set_title(" ".join("270"))

        plt.savefig(output_filename)
        plt.close()

    print(ori_90)
    print(ori_180)
    print(ori_270)

if __name__ == '__main__':
    main()
