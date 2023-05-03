import os
import shutil

def flatten(src_dir, dest_dir):
    for (root, _, fnames) in os.walk(src_dir):
        for fname in fnames:
            try:
                fpath = os.path.join(root, fname)
                dest_fpath = os.path.join(dest_dir, fname)
                shutil.copyfile(fpath, dest_fpath)
            except:
                print("pass")
                pass
                


def check_counts(src_dir, dest_dir):
    files_reached = []
    for (_, _, fnames) in os.walk(src_dir):
        for fname in fnames:
            try:
                files_reached.append(fname)
            except:
                pass
    print(f"{len(set(files_reached))} files reached")
    print(f"{len(os.listdir(dest_dir))} files copied")

    if len(set(files_reached)) == len(os.listdir(dest_dir)):
        print("All files present")
    else:
        print("File count mismatch")



TRAIN_PATH = "/Users/nara/personal/datasets/imagenet-mini/train_/"
VAL_PATH = "/Users/nara/personal/datasets/imagenet-mini/val_/"

TRAIN_FOLDER = "/Users/nara/personal/datasets/imagenet-mini/train/"
VAL_FOLDER = "/Users/nara/personal/datasets/imagenet-mini/val/"

print("Flattening folders...")
flatten(TRAIN_PATH, TRAIN_FOLDER)
flatten(VAL_PATH, VAL_FOLDER)

print("Checking...")
check_counts(TRAIN_PATH, TRAIN_FOLDER)
check_counts(VAL_PATH, VAL_FOLDER)

