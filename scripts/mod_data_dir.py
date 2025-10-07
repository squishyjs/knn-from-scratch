import os
import shutil

base_dir = "./data"

# loop through subdirectories 0–9
for i in range(10):
    outer_dir = os.path.join(base_dir, str(i))
    inner_dir = os.path.join(outer_dir, str(i))

    # check if the nested directory exists
    if os.path.isdir(inner_dir):
        for file_name in os.listdir(inner_dir):
            src_path = os.path.join(inner_dir, file_name)
            dst_path = os.path.join(outer_dir, file_name)

            # only move PNG files
            if file_name.lower().endswith(".png"):
                print(f"Moving {src_path} → {dst_path}")
                shutil.move(src_path, dst_path)

        # remove the now-empty inner folder
        print(f"Removing directory: {inner_dir}")
        os.rmdir(inner_dir)

print("✅ Done! Directory structure flattened.")
