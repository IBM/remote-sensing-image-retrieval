# Remote sensing image retrieval

This is the official implementation of the paper **Multi-Spectral Remote Sensing Image Retrieval using Geospatial Foundation Models**. 
The experiments are explained in our [paper](https://arxiv.org/abs/2403.02059), and you find more information about Prithvi on [Hugging Face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M).

## Approach

![approach.png](figures%2Fapproach.png)

GeoFM embeddings enable simple but accurate content-based image retrieval of remote sensing images. Optionally, the embeddings are compressed into smaller binary vectors to speed up the process and reduce memory usage. 
For each query image, similar images from the database are returned and sorted based on their distance to the query embedding.

## Experimental results

![experimental_results.png](figures%2Fexperimental_results.png)

The table presents the mAP@20 results for all evaluated models and datasets. We highlight the best-performing method in bold and underline the second-best one. LSH is reported with a 95% confidence interval based on five seeds as this method uses random hyperplanes.


![examples.png](figures%2Fexamples.png)

The figure displays examples from two datasets with query images (left), their labels, and retrieved images (right) using Prithvi and the trivial hash method. Images with green frames indicate positive matches, while those with red frames have different labels. Orange shows partial correct matches, where the number represents the number of label matches within the multi-labels.

## Setup

Create an environment and install the required packages with:
```sh
# create env
conda create -n "rsir" python=3.10
# activate env
conda activate rsir
# install pytorch (e.g., with CUDA 12.1, see https://pytorch.org for other versions)
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
# install requirements
pip install -r requirements.txt
```

### Datasets

We provide bash script for downloading the datasets. Run the following script from the project root:

```shell
# optionally specific the data directory (default: 'data/')
export DATA_DIR=data

# Download BigEarthNet (~66 Gb, ~1h)
sh datasets/bigearthnet_download.sh

# Download ForestNet (3 Gb, ~5 min)
sh datasets/forestnet_download.sh
```

### Models

You can download the model weights for Prithvi-100M from [Hugging Face](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/blob/main/Prithvi_100M.pt) with the following commands.

```shell
mkdir weights
cd weights && wget https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M/resolve/main/Prithvi_100M.pt
```

The weights are saved at `weights/Prithvi_100M.pt` but you can also update the path in the config file `configs/prithvi_vit_us.yaml`.

The weights for the vanilla ViT with RGB channels are downloaded automatically. 


## Run experiments

You can save the embeddings of a dataset with:
```shell
# Save embeddings
python inference.py -c configs/prithvi_vit.yaml --dataset ForestNet --split val
python inference.py -c configs/prithvi_vit.yaml --dataset ForestNet --split test
```

If you want to save the embeddings of all evaluated models and dataset versions, you can run:
```shell
bash inference.sh
```

Evaluate all saved embeddings with given 
```shell
# Run experiments
python experiments.py --match any --distance_function hamming --hash_method trivial --hash_length 32
# You can also combine multiple methods
python experiments.py --match any --distance_function hamming --hash_method trivial,lsh,none --hash_length 32,768
```


### Speed experiments

You need a running [Milvus](https://milvus.io) instance for these experiments.

With saved BigEarthNet embeddings, run the experiments with:
```shell
python speed_test_milvus.py
```

If you want to run the experiments on another machine, connect to Milvus via ssh. 

```shell
ssh <server> -L19530:localhost:19530
```

## Citation

```text
@article{RSIR2024,
  title={{Multi-Spectral Remote Sensing Image Retrieval using Geospatial Foundation Models}},
  author={Blumenstiel, Benedikt and Moor, Viktoria and Kienzler, Romeo and Brunschwiler, Thomas},
  journal={arXiv preprint arXiv:2403.02059},
  year={2024}
}
```
