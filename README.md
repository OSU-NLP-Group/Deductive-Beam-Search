<h1 align="center">Deductive Beam Search<br>Decoding Deducible Rationale for Chain-of-Thought Reasoning </h1>

![Static Badge](https://img.shields.io/badge/task-reasoning-purple)
<!-- ![GitHub Repo stars](https://img.shields.io/github/stars/OSU-NLP-Group/Deductive-Beam-Search) -->

Source Code of paper [`Deductive Beam Search: Decoding Deducible Rationale for Chain-of-Thought Reasoning`]().

# Release Progress
- Data
  - [x] Training Data

- Code
  - [x] Training Codes
  - [x] Inference Codes
  - [ ] Evaluation Codes
  - [ ] Dataset Construction Codes



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
```bash
python src/train.py \
     --train_datapath your_data_path \
     --experiment_name your_experiment_name \
     --batch_size 8 --gradient_accumulation_steps 16
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

# Contact

If you have any problems, please contact 
[Tinghui Zhu](mailto:darthzhu@gmial.com) and
[Kai Zhang](mailto:zhang.13253@osu.edu).

# Citation Information

If you find our codes and data useful, please consider citing our paper:

```
@article{zhu2024deductive
  title={Deductive Beam Search: Decoding Deducible Rationale for Chain-of-Thought Reasoning},
  author={Tinghui Zhu and Kai Zhang and Jian Xie and Yu Su},
  journal={arXiv preprint},
  year={2024}
}
```