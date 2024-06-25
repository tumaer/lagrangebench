"""Count the number of files in each subdirectory of a direcory. For sanity check."""

import argparse
import os


def count_files(src, target_count):
    dirs = os.listdir(src)
    dirs.sort()
    for dir in dirs:
        files = os.listdir(os.path.join(src, dir))
        fail_message = "Failed!" if len(files) != target_count else ""
        print(f"In {dir} there are {len(files)} files. {fail_message}")

    print(f"Total number of dirs: {len(dirs)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count the number of files in each subdirectory of a direcory."
    )
    parser.add_argument("--src_dir", type=str, help="Source directory.")
    parser.add_argument("--target_count", type=int, help="Expected number of files.")
    args = parser.parse_args()

    count_files(args.src_dir, args.target_count)

    # example:
    # python count_files.py --src_dir=raw/2D_TGV_2500_10kevery100 --target_count=402
