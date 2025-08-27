# Prepare Datasets for AutoQ-VIS

The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  imagenet/
  ytvis_2019/
```

Expected dataset structure for [YouTubeVIS 2019](https://competitions.codalab.org/competitions/20128):
```
ytvis_2019/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

Please check expected dataset structure for ImageNet-1K at [here](../../datasets/README.md). You can directly [download](https://drive.google.com/file/d/1gllHvrZQNVXphnk-IQxMcXh87Qs86ofT/view?usp=sharing) the pre-processed ImageNet-1K annotations produced by MaskCut in YouTubeVIS format and place it under the "imagenet/annotations/" directory.

Since AutoQ-VIS is class-agnostic, you need to preprocess the YouTubeVIS-2019 dataset using script `prepare_ytvis.py`

To add your own dataset, modify `mask2former_video/data_video/datasets/builtin.py` to make sure the paths of the dataset file are correctly set
