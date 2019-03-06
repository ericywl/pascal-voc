import pandas as pd
import os
#from bs4 import BeautifulSoup
#from more_itertools import unique_everseen
import numpy as np
#import matplotlib.pyplot as plt
#import skimage
#from skimage import io


class PascalVOC:
    """
    Handle Pascal VOC dataset
    """

    def __init__(self, root_dir):
        """
        Summary:
            Init the class with root dir
        Args:
            root_dir (string): path to your voc dataset
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'JPEGImages/')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        self.cache_dir = os.path.join(root_dir, 'csvs')
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def list_image_sets(self):
        """
        Summary:
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    def _imgs_from_category(self, cat_name, dataset):
        """
        Summary:
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = os.path.join(
            self.set_dir, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        return df

    def imgs_from_category_as_list(self, cat_name, dataset):
        """
        Summary:
            Get a list of filenames for images in a particular category
            as a list rather than a pandas dataframe.
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            list of srings: all filenames from that category
        """
        df = self._imgs_from_category(cat_name, dataset)
        df = df[df['true'] == 1]
        return df['filename'].values

    def imgs_as_df(self, dataset):
        df = None
        for cat_name in self.list_image_sets():
            if df is None:
                df = pv._imgs_from_category(cat_name, dataset)
                df = df.replace({-1: 0})
            else:
                new_col = pv._imgs_from_category(cat_name, dataset)
                new_col = new_col.replace({-1: 0})
                df = pd.merge(df, new_col, on='filename')
            df.rename(columns={'true': cat_name}, inplace=True)
        return df

    def imgs_to_fnames_labels(self, dataset):
        df = self.imgs_as_df(dataset)
        filenames = df['filename'].tolist()
        labels = df.drop(['filename'], axis=1).values.tolist()
        return filenames, labels

if __name__ == '__main__':
    pv = PascalVOC('./VOCdevkit/VOC2012/')
    dataset = 'trainval'
    print(df.head())
