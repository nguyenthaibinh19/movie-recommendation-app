"""Validate the checked-in inference artifacts; does not retrain or evaluate."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'Data' / 'artifacts'


def check():
    names = ['deepfm_pytorch.pth', 'item_emb.npy', 'item_id_to_index.json',
             'index_to_item_id.json', 'popular_items.json']
    missing = [name for name in names if not (ART / name).is_file()]
    if missing:
        raise ValueError(f'Missing artifacts: {missing}')
    forward = json.loads((ART / 'item_id_to_index.json').read_text())
    reverse = json.loads((ART / 'index_to_item_id.json').read_text())
    count = len(forward)
    if set(forward.values()) != set(range(count)) or len(reverse) != count:
        raise ValueError('Item mapping is not a contiguous bijection')
    if any(str(reverse.get(str(index))) != item for item, index in forward.items()):
        raise ValueError('Forward and reverse item mappings disagree')
    state = torch.load(ART / 'deepfm_pytorch.pth', map_location='cpu', weights_only=True)
    embeddings = np.load(ART / 'item_emb.npy', allow_pickle=False)
    if embeddings.shape != (count, 16) or not np.isfinite(embeddings).all():
        raise ValueError('Expected a finite item embedding matrix of shape (items, 16)')
    if not np.array_equal(embeddings, state['emb.item_id.weight'].numpy()):
        raise ValueError('Retrieval embeddings differ from the DeepFM checkpoint')
    for field, size in {'gender': 2, 'age': 7, 'occupation': 21, 'item_id': count}.items():
        if tuple(state[f'emb.{field}.weight'].shape) != (size, 16):
            raise ValueError(f'Unexpected embedding shape for {field}')
    popular = json.loads((ART / 'popular_items.json').read_text())
    if len(popular) != len(set(popular)) or any(str(item) not in forward for item in popular):
        raise ValueError('Popularity list contains duplicate or unmapped items')
    files = {name: {'bytes': (ART / name).stat().st_size,
                    'sha256': hashlib.sha256((ART / name).read_bytes()).hexdigest()}
             for name in names}
    return {'items': count, 'embedding_dimensions': 16,
            'checkpoint_parameters': sum(value.numel() for value in state.values()),
            'retrieval_embeddings_match_checkpoint': True,
            'popular_items': len(popular), 'files': files}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, help='Optionally write a JSON artifact report')
    args = parser.parse_args()
    report = check()
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report, indent=2))
