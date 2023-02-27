# From Disentangled Representation to Concept Ranking: Interpreting Deep Representations in Image Classification tasks - Paper Code
## General Aspects
The codes presented here were developed for the XKDD-2022 workshop paper. To execute the methods presented here, the NetDissect code must be executed first. The project link is https://github.com/CSAILVision/NetDissect-Lite. I ran it using the "settings.py" file presented here.

In this paper we used the Action40 dataset - http://vision.stanford.edu/Datasets/40actions.html - to extract the features.

The codes "feature_extraction_global.py" and "feature_extraction_local.py" present the general idea about how I extracted global and local concepts from the model, based on the NetDissection result.

The "linear_classification.py" shows how the ranking based on the classification task was developed.

The jupyter notebook "XKDD_metrics.ipynb" presents the metric calculation for the paper.

## Reproducibility
The structure for the reproducibility is:
- Run NetDissect (https://github.com/CSAILVision/NetDissect-Lite) using the "settings.py"
- Download the Action40 dataset - http://vision.stanford.edu/Datasets/40actions.html
- Run "feature_extraction_global.py" to extract the global concepts.
- Run "linear_classification.py" to generate the ranked global concepts.
- Run "feature_extraction_local.py" to extract the local concepts and generate the ranked local concepts.
- Run "XKDD_metrics.ipynb" to analyse the result.

## Reference

@inproceedings{ferreira2023disentangled,
  title={From Disentangled Representation to Concept Ranking: Interpreting Deep Representations in Image Classification Tasks},
  author={Ferreira dos Santos, Eric and Mileo, Alessandra},
  booktitle={Machine Learning and Principles and Practice of Knowledge Discovery in Databases: International Workshops of ECML PKDD 2022, Grenoble, France, September 19--23, 2022, Proceedings, Part I},
  pages={322--335},
  year={2023},
  organization={Springer}
}
