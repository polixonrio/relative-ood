"""Canonical detector method catalog."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MethodParams:
    """Dataset-specific hyperparameters loaded for one detector method."""

    method_name: str
    dataset_name: str
    args: dict[str, object]

    def get(self, name: str, default: object = None) -> object:
        return self.args.get(name, default)

    def get_for_checkpoint(self, checkpoint_name: str, name: str, default: object = None) -> object:
        overrides = OPENOOD_CHECKPOINT_TUNED_PARAMS.get((self.dataset_name, self.method_name, checkpoint_name), {})
        return overrides.get(name, self.get(name, default))


METHODS: dict[str, dict[str, str]] = {
    'msp': {'postprocessor': 'msp', 'label': 'MSP', 'family': 'output'},
    'mls': {'postprocessor': 'mls', 'label': 'MaxLogit', 'family': 'output'},
    'ebo': {'postprocessor': 'ebo', 'label': 'Energy', 'family': 'output'},
    'gen': {'postprocessor': 'gen', 'label': 'GEN', 'family': 'output'},
    'tss': {'postprocessor': 'tss_raw', 'label': 'TSS', 'family': 'output'},
    'rtss': {'postprocessor': 'rtss', 'label': 'rTSS', 'family': 'output'},
    'knn': {'postprocessor': 'knn', 'label': 'KNN', 'family': 'distance'},
    'rknnpp': {'postprocessor': 'rknnpp', 'label': 'RkNN++', 'family': 'distance'},
    'mds': {'postprocessor': 'mds', 'label': 'Mahalanobis', 'family': 'distance'},
    'mdspp': {'postprocessor': 'mdspp', 'label': 'MDS++', 'family': 'distance'},
    'rmds': {'postprocessor': 'rmds', 'label': 'RMDS', 'family': 'distance'},
    'rmdspp': {'postprocessor': 'rmdspp', 'label': 'RMDS++', 'family': 'distance'},
    'lsm': {'postprocessor': 'lsm', 'label': 'LSM', 'family': 'distance'},
    'lsmpp': {'postprocessor': 'lsmpp', 'label': 'LSM++', 'family': 'distance'},
    'rlsm': {'postprocessor': 'rlsm', 'label': 'rLSM', 'family': 'distance'},
    'rlsmpp': {'postprocessor': 'rlsmpp', 'label': 'rLSM++', 'family': 'distance'},
    'react': {'postprocessor': 'react', 'label': 'ReAct', 'family': 'feature_activation'},
    'ash': {'postprocessor': 'ash', 'label': 'ASH', 'family': 'feature_activation'},
    'dice': {'postprocessor': 'dice', 'label': 'DICE', 'family': 'feature_activation'},
    'scale': {'postprocessor': 'scale', 'label': 'SCALE', 'family': 'feature_activation'},
    'vim': {'postprocessor': 'vim', 'label': 'VIM', 'family': 'feature_activation'},
    'gradnorm': {'postprocessor': 'gradnorm', 'label': 'GradNorm', 'family': 'gradient'},
}
METHOD_FAMILY = {spec['label']: spec['family'] for spec in METHODS.values()}
METHOD_ORDER = list(METHOD_FAMILY.keys())
FAMILY_ORDER = ['output', 'distance', 'feature_activation', 'gradient']
FAMILY_COLORS = {
    'output': '#2f6bff',
    'distance': '#13866b',
    'feature_activation': '#cf5c36',
    'gradient': '#8a4fff',
}

OPENOOD_TUNED_PARAMS: dict[tuple[str, str], dict[str, object]] = {
    ('cifar100', 'react'): {'percentile': 99},
    ('imagenet200', 'knn'): {'K': 500},
    ('imagenet', 'react'): {'percentile': 95},
}

OPENOOD_CHECKPOINT_TUNED_PARAMS: dict[tuple[str, str, str], dict[str, object]] = {
    ('cifar100', 'ash', 's0'): {'percentile': 95},
    ('cifar100', 'ash', 's1'): {'percentile': 90},
    ('cifar100', 'ash', 's2'): {'percentile': 90},
    ('imagenet200', 'ash', 's0'): {'percentile': 90},
    ('imagenet200', 'ash', 's1'): {'percentile': 95},
    ('imagenet200', 'ash', 's2'): {'percentile': 95},
    ('imagenet200', 'react', 's0'): {'percentile': 95},
    ('imagenet200', 'react', 's1'): {'percentile': 99},
    ('imagenet200', 'react', 's2'): {'percentile': 99},
}


def method_panel() -> list[dict[str, str]]:
    """Return the fixed thesis method panel in registry order."""
    return list(METHODS.values())


def method_label(method: str) -> str:
    """Return the display label for one canonical method name."""
    return str(METHODS[method]['label'])


def load_method_params(method_name: str, config_root: Path, dataset_name: str) -> MethodParams:
    """Load method parameters from OpenOOD YAML, then apply thesis overrides."""
    config_path = Path(config_root) / 'postprocessors' / f'{method_name}.yml'
    args: dict[str, object] = {}
    if config_path.is_file():
        with config_path.open('r', encoding='utf-8') as handle:
            config = yaml.safe_load(handle) or {}
        args = dict((config.get('postprocessor') or {}).get('postprocessor_args') or {})
    args.update(OPENOOD_TUNED_PARAMS.get((dataset_name, method_name), {}))
    return MethodParams(method_name=method_name, dataset_name=dataset_name, args=args)
