
import os
import argparse
import textwrap
import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from experiments import get_similarity, get_hash


def plot_retrival_results(queries, results, correct, labels):
    """
    Create figure with label names, queries and retrieved images.

    :param queries: tensor of shape [sample, channel, h, w]
    :param results: tensor of shape [sample, images, channel, h, w]
    :param correct: tensor of shape [sample, images] with bool values of retrieved image is correct
    :param labels: list of shape [sample] with label names of each sample
    """
    num_samples, num_images = results.shape[:2]

    # Scale by max sensor value per query
    scale_max = torch.max(torch.amax(queries, dim=(1, 2, 3)), torch.amax(results, dim=(1, 2, 3, 4))) * 0.5
    # Convert all tensors to RGB format
    queries_rgb = (queries / scale_max[:, None, None, None] * 255).to(int).permute(0, 2, 3, 1)
    results_rgb = (results / scale_max[:, None, None, None, None] * 255).to(int).permute(0, 1, 3, 4, 2)
    queries_rgb = queries_rgb.clip(0, 255)
    results_rgb = results_rgb.clip(0, 255)

    # Create subplots
    fig, axs = plt.subplots(num_samples, num_images + 2, figsize=(14, num_samples * 1.2),
                            gridspec_kw={'width_ratios': [1, 2.] + [1] * num_images})
    fig.subplots_adjust(hspace=0.1, wspace=0.1, left=0.01, right=0.99, top=0.99, bottom=0.01)

    for i in range(num_samples):
        # Display query image
        axs[i, 0].imshow(queries_rgb[i])
        axs[i, 0].axis('off')
        # Label to string
        # label_str = ',\n'.join([textwrap.shorten(l, width=30, placeholder='...') for l in labels[i]])
        label_str = ',\n'.join(labels[i])
        # Replace a very long label from BigEarthNet
        label_str = label_str.replace("Land principally occupied by agriculture, "
                                      "with significant areas of natural vegetation",
                                      "Agriculture with\n natural vegetation")
        axs[i, 1].text(0.5, 0.5, label_str, ha='center', va='center', fontsize=10)
        axs[i, 1].axis('off')

        for j in range(num_images):
            # Display result image
            axs[i, j + 2].imshow(results_rgb[i, j])
            axs[i, j + 2].tick_params(left=False, right=False, labelleft=False,
                                      labelbottom=False, bottom=False)
            # Add frame based on correctness
            if correct[i, j] == len(labels[i]):
                frame_color = 'green'
            elif correct[i, j]:
                frame_color = 'orange'
                # Add number of correct labels
                axs[i, j + 2].text(200,  200, str(correct[i, j].item()), fontsize=10,
                                   color='white', ha='center', va='center')
            else:
                frame_color = 'red'
            for spine in axs[i, j + 2].spines.values():
                spine.set_edgecolor(frame_color)
                spine.set_linewidth(3)

    # axs[0, 0].set_title("Query")
    # axs[0, 1 + round(num_images / 2)].set_title("Retrieved images")
    # plt.tight_layout()


def main(args):
    # Init dataset
    with open(args.config_file, 'r') as f:
        # Load config file
        cfg = yaml.safe_load(f)
    cfg['dataset']['name'] = args.dataset_visual
    cfg['dataset']['split'] = 'val'
    val_dataset = load_dataset(cfg)
    cfg['dataset']['split'] = 'test'
    test_dataset = load_dataset(cfg)
    
    # Init embeddings with shape [sample, embedding]
    output_path = os.getenv('OUTPUT_PATH', os.path.join('output', 'embeddings'))
    val_embeddings = torch.load(os.path.join(output_path, cfg['model']['name'], args.dataset, 'val', 'embeddings.pt'),
                                  map_location='cpu')
    test_embeddings = torch.load(os.path.join(output_path, cfg['model']['name'], args.dataset, 'test', 'embeddings.pt'),
                                 map_location='cpu')
    # Multi-labels with shape [sample, label]
    val_labels = torch.load(os.path.join(output_path, cfg['model']['name'], args.dataset, 'val', 'labels.pt'),
                              map_location='cpu')
    test_labels = torch.load(os.path.join(output_path, cfg['model']['name'], args.dataset, 'test', 'labels.pt'),
                              map_location='cpu')

    # Load label names
    if 'BigEarthNet19' in args.dataset:
        label_names = val_dataset.class_sets[19]
    elif 'BigEarthNet' in args.dataset:
        label_names = val_dataset.class_sets[43]
    elif 'ForestNet' in args.dataset:
        label_names = val_dataset.label_names

    # Select sample queries
    np.random.seed(42)
    indices = args.indices or np.random.choice(range(len(val_dataset)), args.num_queries)
    print('Selected indices:', indices)

    # Retrival
    val_hash, test_hash = get_hash([val_embeddings, test_embeddings], 'trivial', length=32)
    similarity = get_similarity(val_hash[indices], test_hash, distance='hamming')
    # Select top k results
    ranking = similarity.topk(args.num_samples, sorted=True, dim=-1)[1]

    # Get correct match
    correct = val_labels[indices].unsqueeze(1).repeat(1, len(test_labels), 1)
    # count label matches
    correct = correct.logical_and(correct == test_labels).sum(dim=-1)
    correct = torch.gather(correct, 1, ranking)

    # Get images
    queries = []
    retrieved_images = []
    labels = []
    for i, r in zip(indices, ranking):
        queries.append(val_dataset[i]['image'])
        retrieved_images.append(torch.stack([test_dataset[n]['image'] for n in r]))
        label_idx = val_labels[i].nonzero().flatten()
        labels.append([label_names[l] for l in label_idx])
    queries = torch.stack(queries)
    retrieved_images = torch.stack(retrieved_images)

    plot_retrival_results(queries, retrieved_images, correct, labels)

    if os.path.isdir(args.output_dir):
        output_file = os.path.join(args.output_dir, f"{cfg['model']['name']}_{args.dataset}_retrieval.pdf")
    else:
        output_file = args.output_dir

    plt.savefig(output_file)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_file', type=str, default='configs/prithvi_vit_us.yaml',
                        help='Path to config file')
    parser.add_argument('-o', '--output_dir', type=str, default='output/figures',
                        help='Path to output dir')
    parser.add_argument('-d', '--dataset', type=str, default='ForestNet')
    parser.add_argument('-v', '--dataset_visual', type=str, default='ForestNetVisual')
    parser.add_argument('-i', '--indices', nargs='+', type=str, default=[84, 435])
    parser.add_argument('-n', '--num_samples', type=int, default=9)
    parser.add_argument('-q', '--num_queries', type=int, default=4)
    args = parser.parse_args()

    main(args)
