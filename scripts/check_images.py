"""
Script to check and visualize sample images from dataset
This helps diagnose issues with image format, color inversion, etc.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from src.dataio import load_image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_sample_images(data_dir, n_samples=5):
    """
    Load and display sample images from each class to check format.

    Args:
        data_dir: str, path to data directory
        n_samples: int, number of samples per class to check
    """
    # read data dir (digits 0 - 9, only numeric classes)
    all_dirs = [d for d in os.listdir(data_dir)
                if os.path.isdir(os.path.join(data_dir, d))]
    class_dirs = sorted([d for d in all_dirs if d.isdigit()])

    if not class_dirs:
        print("No numeric class directories found!")
        return

    print(f"Checking classes: {class_dirs}\n")

    fig, axes = plt.subplots(len(class_dirs), n_samples * 2, figsize=(20, 2 * len(class_dirs)))

    for i, class_label in enumerate(class_dirs):
        class_path = os.path.join(data_dir, class_label)
        image_files = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:n_samples]

        for j, img_file in enumerate(image_files):
            img_path = os.path.join(class_path, img_file)

            try:
                # load image
                img = Image.open(img_path)

                print(f"Class {class_label}, Image {j+1}: Mode={img.mode}, Size={img.size}")

                # handle transparency -> WHITE background
                if img.mode in ('RGBA', 'LA', 'PA'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'RGBA':
                        background.paste(img, mask=img.split()[3])
                    elif img.mode == 'LA':
                        background.paste(img, mask=img.split()[1])
                    else:
                        background.paste(img)
                    img = background
                    print(f"  → Converted from transparent to white background")

                # grayscale convert
                if img.mode != 'L':
                    img = img.convert('L')

                img_array = np.array(img)
                mean_val = np.mean(img_array)
                min_val = np.min(img_array)
                max_val = np.max(img_array)

                ax_orig = axes[i, j*2] if len(class_dirs) > 1 else axes[j*2]
                ax_orig.imshow(img_array, cmap='gray', vmin=0, vmax=255)
                ax_orig.set_title(f'Class {class_label}\nMean:{mean_val:.1f}', fontsize=8)
                ax_orig.axis('off')

                # (if needed) display inverted background
                img_inverted = 255 - img_array
                ax_inv = axes[i, j*2+1] if len(class_dirs) > 1 else axes[j*2+1]
                ax_inv.imshow(img_inverted, cmap='gray', vmin=0, vmax=255)
                ax_inv.set_title(f'Inverted\nMean:{np.mean(img_inverted):.1f}', fontsize=8)
                ax_inv.axis('off')

                # print info for first image
                if j == 0:
                    print(f"  After processing:")
                    print(f"    Mean pixel value: {mean_val:.1f}")
                    print(f"    Range: [{min_val}, {max_val}]")
                    if mean_val < 128:
                        print(f"    ⚠️  Images appear DARK (digits darker than background)")
                        print(f"    ✓  Should use invert=True (default)")
                    else:
                        print(f"    ✓  Images appear LIGHT (digits lighter than background)")
                        print(f"    ✓  Should use --no-invert")
                    print()

            except Exception as e:
                print(f"  ✗ Error loading {img_path}: {e}")

    plt.suptitle('Left: With White Background | Right: Inverted\nCheck which shows digits correctly!',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    print("\n" + "="*70)
    print("RECOMMENDATION:")
    print("="*70)
    print("Look at the visualization above:")
    print("- If the LEFT column shows correct black digits on white:")
    print("  → Use: --no-invert")
    print("- If the RIGHT column shows correct digits:")
    print("  → Use default (auto-inverts)")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Check sample images from dataset')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing data')
    parser.add_argument('--n_samples', type=int, default=3,
                        help='Number of samples to check per class')

    args = parser.parse_args()

    print("="*70)
    print("IMAGE FORMAT CHECKER")
    print("="*70)
    print(f"Checking images in: {args.data_dir}\n")

    check_sample_images(args.data_dir, args.n_samples)


if __name__ == "__main__":
    main()