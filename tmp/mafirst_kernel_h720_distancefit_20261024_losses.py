"""Frozen per-channel train-only scales; legacy Gram diagnostics under no_grad."""
import json
from pathlib import Path
import torch
from utils.patch_distance_huber_20261024 import TERM_NAMES,distance_parts as _distance_parts
from utils.global_joint_output_gradients_20261024 import gradient_summary
from utils.drop_output_gram_mse_20261024 import training_terms as diagnostic_terms,output_parts
R=Path(__file__).resolve().parents[1]
SCALE_FILE=R/'tmp/mafirst_kernel_h720_distancefit_group_20261024/distance_scale.json'
SCALES=tuple(json.loads(SCALE_FILE.read_text())['scales'])
def distance_parts(prediction,target,head):return _distance_parts(prediction,target,head,SCALES)
def training_terms(prediction,target,head,relation_mode):
    assert relation_mode=='none'
    distances=distance_parts(prediction,target,head)
    with torch.no_grad():old,details=diagnostic_terms(prediction.detach(),target.detach(),head.detach(),'none')
    details['legacy_terms']={n:v.detach() for n,v in old.items()};details['distance']=distances
    terms=dict(point_mse=(prediction-target).square().mean(),head_distance_huber=distances['head_huber'],output_distance_huber=distances['output_huber'])
    assert tuple(terms)==TERM_NAMES
    return terms,details
