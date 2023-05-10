# AlignSTS: Speech-to-Singing Conversion via Cross-Modal Alignment

#### Ruiqi Li, Rongjie Huang, Lichao Zhang, Jinglin Liu, Zhou Zhao | Zhejiang University

PyTorch Implementation of [AlignSTS (ACL 2023)](https://arxiv.org/abs/2305.04476): a speech-to-singing (STS) model based on modality disentanglement and cross-modal alignment.

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2305.04476)

We provide our implementation and pretrained models in this repository.

Visit our [demo page](https://alignsts.github.io/) for audio samples.

## News
- May, 2023: AlignSTS released at Github.
- May, 2023: AlignSTS accepted at ACL 2023 Findings.

## Quick Start
We provide an example of how you can generate high-quality samples using AlignSTS.

### Pretrained Models
You can use pretrained models we provide. Details of each folder are as in follows:

| Model       | Discription                                                              | 
|-------------|--------------------------------------------------------------------------|
|   AlignSTS  | Acousitic model [(config)](configs/singing/speech2singing/alignsts.yaml) |
| HIFI-GAN    | Neural Vocoder                                                           |

### Dependencies

A suitable [conda](https://conda.io/) environment named `alignsts` can be created and activated with:

```
conda env create -f environment.yaml
conda activate alignsts
```

### Test samples

We provide a mini-set of test samples to demonstrate AlignSTS. Specifically, we provide samples of WAV format combining the corresponding statistical files which is for faster IO. Please download the statistical files at `data/binary/speech2singing-testdata/`, while the WAV files are for listening.

FYI, the naming rule of the WAV files is `[spk]#[song name]#[speech/sing identifier]#[sentence index].wav`. For example, a sample named `男3号#all we know#sing#14.wav` means a singing sample of song "all we know" from the 14th sentence, sung by the speaker "男3号".

### Inference
Here we provide a speech-to-singing conversion pipeline using AlignSTS. 

1. Prepare **AlignSTS** (acoustic model): Download and put checkpoint at `checkpoints/alignsts`
2. Prepare **HIFI-GAN** (neural vocoder): Download and put checkpoint at `checkpoints/hifigan`
3. Prepare **dataset** (test dataset): Download the statistical files of the test dataset at `data/binary/speech2singing-testdata`
4. Run
```bash
CUDA_VISIBLE_DEVICES=0 python tasks/run.py --exp_name alignsts --infer --hparams "gen_dir_name=test" --config configs/singing/speech2singing/alignsts.yaml --reset
```
5. You will find outputs in `checkpoints/alignsts/generated_200000_test`, where [G] indicates ground truth mel results and [P] indicates predicted results.

## Acknowledgements
This implementation uses parts of the code from the following Github repos:
[NATSpeech](https://github.com/NATSpeech/NATSpeech),
[ProDiff](https://github.com/Rongjiehuang/ProDiff),
[SpeechSplit2](https://github.com/biggytruck/SpeechSplit2)
as described in our code.

## Citations ##
If you find this code useful in your research, please cite our work:
```bib
@article{li2023alignsts,
  title={AlignSTS: Speech-to-Singing Conversion via Cross-Modal Alignment},
  author={Li, Ruiqi and Huang, Rongjie and Zhanag, Lichao and Liu, Jinglin and Zhao, Zhou},
  journal={Association for Computational Linguistics},
  year={2023}
}
```

## Disclaimer ##
Any organization or individual is prohibited from using any technology mentioned in this paper to generate someone's speech/singing without his/her consent, including but not limited to government leaders, political figures, and celebrities. If you do not comply with this item, you could be in violation of copyright laws.

