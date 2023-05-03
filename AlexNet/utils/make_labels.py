import os
import pandas as pd

"""
This module is hard coded to the dataset location and coded around the
mini-imagenet datasets. Use with care!

The goal is to create a csv file with idx 0 as the image filename and 
idx 1 as labels
"""

DATASET_PATH = "/Users/nara/personal/datasets/imagenet-mini"
TRAIN_PATH = os.path.join(DATASET_PATH, "train")
VAL_PATH = os.path.join(DATASET_PATH, "val")


def make_labels(dir, output_fname):
    df = pd.DataFrame(columns=["fname", "label"])
    folders = os.listdir(dir)

    for folder in folders:
        folder_dir = os.path.join(dir, folder)
        print(folder_dir)
        try:
            fnames = os.listdir(folder_dir)
        except:
            pass
        for fname in fnames:
            d = {"fname": [fname], "label": [folder]};
            df2 = pd.DataFrame(data=d)
            df = pd.concat([df, df2])

    df.to_csv(output_fname, index=False)

if __name__ == "__main__":
    print("Creating annotations file for train and val datasets")
    make_labels(TRAIN_PATH, "train_labels.csv")
    make_labels(VAL_PATH, "val_labels.csv")
    print("Files created")
    
