<h1 align="center">Deductive Beam Search<br>Decoding Deducible Rationale for Chain-of-Thought Reasoning </h1>

![Static Badge](https://img.shields.io/badge/task-reasoning-purple)
<!-- ![GitHub Repo stars](https://img.shields.io/github/stars/OSU-NLP-Group/Deductive-Beam-Search) -->

Source Code of paper [`Deductive Beam Search: Decoding Deducible Rationale for Chain-of-Thought Reasoning`](https://arxiv.org/abs/2401.17686).

# Quick Start
Clone repo:
```bash
git clone https://github.com/OSU-NLP-Group/Deductive-Beam-Search.git
cd Deductive-Beam-Search
mkdir outputs
```

Install requirements:
```bash
conda create --name dbs python=3.10
conda activate dbs
pip install -r requirements.txt
```

Training:

- Stage 1:

  ```bash
  python src/train.py \
      --train_datapath your_data_path_stage1 \
      --experiment_name your_experiment_name_stage1 \
      --batch_size 8 --gradient_accumulation_steps 16 \
      --learning_rate 1e-5
  ```

- Stage 2:

  ```bash
  python src/train.py \
      --train_datapath your_data_path_stage2 \
      --experiment_name your_experiment_name_stage2 \
      --batch_size 8 --gradient_accumulation_steps 16 \
      --learning_rate 1e-7
  ```

Inference:
```bash
DECODER_PATH=your_model_path
DATASETS=("gsm8k" "svamp") # add datasets you want to test

for TEST_DATASET in ${DATASETS[@]};
do
    python src/test.py \
        --test_dataset $TEST_DATASET \
        --output_dir outputs/$TEST_DATASET \
        --decoder_path $DECODER_PATH \
        --decode_strategy beam_search \
        --num_beams 5 \
        --num_sampling 10 \
        --num_gpus_decode 4 # how many gpus for inference
done
```

# Data

## Training Data

All training data synthesized is in `data/` folder. The training file `data/stage1/train.json` is used for training a general deductive verifier. For stage 2, the arithmetic and symbolic verifier is trained on `data/stage2/arithmetic/train.split.*.json`, and the commonsense verifier is trained on `data/stage2/commonsense/train.json`.

## Checkpoints

We provide the checkpoint of a general deductive verifier, please download from this [link](https://drive.google.com/drive/folders/1GbnAiX160Cz63zAbr2FAgB0QFySfM2Vn?usp=sharing).
You can use it to continue-train on our data or train on your own data.

## Data Generation
The complete process of data construction is in `src/generate_data.py`.
If you want to generate data on your own, please modify the data loading part.
After modifying, you can run `python src/generate_data.py` to generate data for your own domains.

**\[TODO\]** We will improve the code for easier modification and usage.

# Contact

If you have any problems, please contact 
[Tinghui Zhu](mailto:darthzhu@gmail.com) and
[Kai Zhang](mailto:zhang.13253@osu.edu).

# Citation Information

If you find our codes and data useful, please consider citing our paper:

```
@article{zhu2024deductive,
  title={Deductive Beam Search: Decoding Deducible Rationale for Chain-of-Thought Reasoning},
  author={Zhu, Tinghui and Zhang, Kai and Xie, Jian and Su, Yu},
  journal={arXiv preprint arXiv:2401.17686},
  year={2024}
}
```