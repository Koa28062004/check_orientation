import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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

    for img_path in args.input_path.rglob("crop_*.jpg"):
        image = load_rgb(img_path)

        # Convert the NumPy array to a PyTorch tensor
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        with torch.no_grad():
            prediction = model(image_tensor).numpy()

        # Construct output path similar to input path but within the result folder
        relative_path = img_path.relative_to(args.input_path)
        output_path = args.output_path / relative_path.parent
        output_path.mkdir(parents=True, exist_ok=True)
        output_filename = output_path / f"before_{img_path.stem}.jpg"

        print(output_filename)

        # Plot and save the resulting images
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(image)
        ax.axis('off')
        
        orientation = prediction[0]
        max_position = np.argmax(orientation)
        times = 0

        if max_position == 1: 
            ori_270.append(str(img_path)) 
            times = 3
            # ax.set_title("90")
        elif max_position == 2: 
            ori_180.append(str(img_path))
            times = 2 
            # ax.set_title("180")
        elif max_position == 3:
            ori_90.append(str(img_path))  
            times = 1
            # ax.set_title("270")

        plt.imsave(output_filename, image)
        plt.close()

        if times == 0:
            rotated_image = image
        else:
            rotated_image = np.rot90(image, k=times)

        output_rotated_filename = output_path / f"{img_path.stem}.jpg"
        plt.imsave(output_rotated_filename, rotated_image)
        plt.close()

    # Write ori lists to JSON file
    with open(args.output_path / "check_orientation.json", "w") as f:
        json.dump({
            "ori_90": ori_90,
            "ori_180": ori_180,
            "ori_270": ori_270
        }, f, indent=4)

    print("Images rotated by 90 degrees:", ori_90)
    print("Images rotated by 180 degrees:", ori_180)
    print("Images rotated by 270 degrees:", ori_270)

if __name__ == '__main__':
    main()
