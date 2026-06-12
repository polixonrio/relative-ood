"""Fixed benchmark choices used by the thesis analyses."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / 'results'
CACHE_DIR = RESULTS_DIR / 'cache' / 'model'
DATA_ROOT = REPO_ROOT / 'data'
CONFIG_ROOT = REPO_ROOT / 'configs'

ACTIVE_DATASETS = ['cifar100', 'imagenet200', 'imagenet']
CACHE_DEVICE = 'cuda:0'
CACHE_BATCH_SIZE = 128
CACHE_NUM_WORKERS = 0
REBUILD_MODEL_CACHE = False
REBUILD_ANALYSIS_OUTPUTS = True

DATASET_NETWORKS = {
    'cifar100': 'resnet18_32x32',
    'imagenet200': 'resnet18_224x224',
    'imagenet': 'resnet50',
}

DATASET_CHECKPOINT_SOURCES = {
    'cifar100': 'cifar100_resnet18_32x32_base_e100_lr0.1_default',
    'imagenet200': 'imagenet200_resnet18_224x224_base_e90_lr0.1_default',
    'imagenet': 'pretrained_weights/resnet50_imagenet1k_v1.pth',
}

DATASET_PROTOCOLS = {
    'cifar100': ['standard'],
    'imagenet200': ['standard', 'full_spectrum'],
    'imagenet': ['standard', 'full_spectrum'],
}

FULL_SPECTRUM_DATASETS = [dataset for dataset, protocols in DATASET_PROTOCOLS.items() if 'full_spectrum' in protocols]

DATASET_SAMPLE_GROUPS = {
    'cifar100': {
        'clean_id': [('id', 'test')],
        'cs_id': [('csid', 'cifar100c')],
        'near_ood': [('ood_near', 'cifar10'), ('ood_near', 'tin')],
        'far_ood': [('ood_far', 'mnist'), ('ood_far', 'svhn'), ('ood_far', 'texture'), ('ood_far', 'places365')],
    },
    'imagenet200': {
        'clean_id': [('id', 'test')],
        'cs_id': [('csid', 'imagenet_v2'), ('csid', 'imagenet_c'), ('csid', 'imagenet_r')],
        'near_ood': [('ood_near', 'ssb_hard'), ('ood_near', 'ninco')],
        'far_ood': [('ood_far', 'inaturalist'), ('ood_far', 'textures'), ('ood_far', 'openimage_o')],
    },
    'imagenet': {
        'clean_id': [('id', 'test')],
        'cs_id': [('csid', 'imagenet_v2'), ('csid', 'imagenet_c'), ('csid', 'imagenet_r')],
        'near_ood': [('ood_near', 'ssb_hard'), ('ood_near', 'ninco')],
        'far_ood': [('ood_far', 'inaturalist'), ('ood_far', 'textures'), ('ood_far', 'openimage_o')],
    },
}
