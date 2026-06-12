"""Step 00: build model-output caches for fixed thesis benchmarks."""

from __future__ import annotations

import shutil
from types import SimpleNamespace

import torch

from openood.evaluation_api.datasets import DATA_INFO
from openood.evaluation_api.datasets import data_setup, get_id_ood_dataloader
from openood.evaluation_api.preprocessor import get_default_preprocessor
from openood.networks.utils import get_network
from analysis.core.benchmarks import ACTIVE_DATASETS, CACHE_BATCH_SIZE, CACHE_DEVICE, CACHE_DIR, CACHE_NUM_WORKERS, DATA_ROOT, DATASET_CHECKPOINT_SOURCES, DATASET_NETWORKS, DATASET_SAMPLE_GROUPS, REBUILD_MODEL_CACHE, RESULTS_DIR
from analysis.core.model_cache import ModelCache, cache_root_name, write_cache_index


def main() -> None:
    if not CACHE_DEVICE.startswith('cuda'):
        raise ValueError('OpenOOD evaluator currently requires a CUDA device')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required but not available')
    _, _, index = CACHE_DEVICE.partition(':')
    torch.cuda.set_device(int(index) if index else 0)

    records = []

    for dataset in ACTIVE_DATASETS:
        network = DATASET_NETWORKS[dataset]
        checkpoint_source = RESULTS_DIR / DATASET_CHECKPOINT_SOURCES[dataset]
        if checkpoint_source.is_file():
            checkpoint_paths = [checkpoint_source.resolve()]
        elif checkpoint_source.is_dir():
            checkpoint_paths = []
            for directory in sorted(path for path in checkpoint_source.rglob('*') if path.is_dir()):
                best = directory / 'best.ckpt'
                last = directory / 'last.ckpt'
                if best.is_file():
                    checkpoint_paths.append(best.resolve())
                elif last.is_file():
                    checkpoint_paths.append(last.resolve())
        else:
            raise FileNotFoundError(f'Fixed checkpoint source for dataset={dataset} does not exist: {checkpoint_source}')
        if not checkpoint_paths:
            raise FileNotFoundError(f'No best.ckpt or last.ckpt files found under {checkpoint_source}')

        print(f'Preparing dataloaders for {dataset} (batch_size={CACHE_BATCH_SIZE}, num_workers={CACHE_NUM_WORKERS})...', flush=True)
        preprocessor = get_default_preprocessor(dataset)
        data_setup(str(DATA_ROOT), dataset)
        dataloader_dict = get_id_ood_dataloader(
            dataset, str(DATA_ROOT), preprocessor, batch_size=CACHE_BATCH_SIZE, shuffle=False, num_workers=CACHE_NUM_WORKERS
        )
        sample_groups = DATASET_SAMPLE_GROUPS[dataset]
        split_manifest = {
            'id': ['test', 'val', 'train'],
            'csid': [name for _, name in sample_groups['cs_id']],
            'ood_near': [name for _, name in sample_groups['near_ood']],
            'ood_far': [name for _, name in sample_groups['far_ood']],
        }
        splits = [('id', 'test'), ('id', 'val'), ('id', 'train'), *sample_groups['cs_id'], *sample_groups['near_ood'], *sample_groups['far_ood']]
        print(f'[dataset] {dataset} / {network}: {len(checkpoint_paths)} checkpoint(s), {len(splits)} split(s)')

        for checkpoint_path in checkpoint_paths:
            checkpoint_name = checkpoint_path.parent.name if checkpoint_source.is_dir() else checkpoint_path.stem
            checkpoint_stat = checkpoint_path.stat()
            metadata = {
                'checkpoint_name': checkpoint_name,
                'checkpoint_path': str(checkpoint_path),
                'checkpoint_size': int(checkpoint_stat.st_size),
                'checkpoint_mtime_ns': int(checkpoint_stat.st_mtime_ns),
                'data_root': str(DATA_ROOT.resolve()),
                'dataset': dataset,
                'network': network,
            }
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            dataset_info = DATA_INFO[dataset]
            network_config = SimpleNamespace(name=network, num_classes=dataset_info['num_classes'], pretrained=False, checkpoint='', num_gpus=0)
            net = get_network(network_config)
            net.load_state_dict(state_dict, strict=True)
            net.cuda()
            net.eval()
            cache_root = CACHE_DIR / cache_root_name(metadata)
            if REBUILD_MODEL_CACHE and cache_root.exists():
                shutil.rmtree(cache_root)
            model_cache = ModelCache(cache_root, net=net, dataloader_dict=dataloader_dict, metadata={**metadata, 'splits': split_manifest})
            print(f'[cache] {metadata["checkpoint_name"]} -> {model_cache.cache_root}')
            for group, split_name in splits:
                model_cache.get_outputs(group, split_name)
            records.append({**metadata, 'cache_root': str(model_cache.cache_root), 'split_count': len(splits)})

    index_path = write_cache_index(records, CACHE_DIR)
    print(f'Wrote cache index to {index_path}')


if __name__ == '__main__':
    main()
