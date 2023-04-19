# LiTS17 [^1]
## Getting the data
1. Download it from [https://competitions.codalab.org/competitions/17094](https://competitions.codalab.org/competitions/17094)

2. Create a new folder `LITS` at the main directory of this project

3. Uncompress `Training_Batch1.zip` and `Training_Batch2.zip` and move all the segmentation and volume NIfTI files to a new folder `train` inside `LITS`.


## Processing LiTS17 Liver dataset
See [docs/create_lits17_liver_dataset.md](docs/create_lits17_liver_dataset.md)
**Note:** The default configuration is employed to process the LITS17 Liver 1 32x80x80-crops dataset in the paper "Disagreement attention: Let us agree to disagree on computed tomography segmentation" [^2].

## Processing LiTS17 Lesion to obtain a 16 32x160x160-crops dataset
All the steps to create a dataset by extracting 16 32x160x160-crops per CT
considering only lesion labels.

See [docs/create_lits17_lesion_16_32x160x160-crops_dataset.md](docs/create_lits17_lesion_16_32x160x160-crops_dataset.md)

**Note:** The default configuration is employed to generate the LITS17 Lesion 16 32x160x160-crops dataset in the paper "Disagreement attention: Let us agree to disagree on computed tomography segmentation" [^2].


## Using LiTS17CropMGR and calculating min_crop_mean
See [docs/calculate_min_crop_mean.md](docs/calculate_min_crop_mean.md)

[^1]: P. Bilic et al., “The liver tumor segmentation benchmark (LiTS),” arXiv e-prints, p. arXiv:1901.04056, Jan. 2019. [Online]. Available: [https://arxiv.org/abs/1901.04056](https://arxiv.org/abs/1901.04056)
[^2]: Lopez Molina, E. G., Huang, X., & Zhang, Q. (2023). Disagreement attention: Let us agree to disagree on computed tomography segmentation. Biomedical Signal Processing and Control, 84, 104769. [https://doi.org/10.1016/j.bspc.2023.104769](https://doi.org/10.1016/j.bspc.2023.104769)
