from argparse import ArgumentParser

def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--experiment_name", default="gsm8k")
    
    parser.add_argument("--train_datapath", default="data/gsm8k/train_ranking_small_full.json")
    parser.add_argument("--valid_datapath", default=None)
    parser.add_argument("--test_datapath", default=None)
    
    parser.add_argument("--save_dir", default="ckpts/gsm8k")
    parser.add_argument("--resume_path", default=None)
    
    parser.add_argument("--verifier_model_path", default="data/cache/deberta/64a8c8eab3e352a784c658aef62be1662607476f")
    parser.add_argument("--hidden_size", type=int, default=1024)

    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--dense_dim", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_warmup_steps", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--lr_step_size", type=int, default=5)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=int, default=0.0005)
    
    parser.add_argument("--min_margin", type=float, default=0.1)
    parser.add_argument("--max_margin", type=float, default=0.3)
    parser.add_argument("--margin_increase_step", type=int, default=2000)

    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--device")
    
    parser.add_argument("--test_dataset", type=str, choices=["gsm8k", "svamp", "aqua", "addsub", "singleeq", "multiarith", "strategyqa", "csqa", "llc", "coin"])
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--decoder_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--num_sampling", type=int, default=1)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--decode_strategy", type=str, default="direct")
    parser.add_argument("--shorten_context", action="store_true")
    parser.add_argument("--is_greedy", action="store_true", default=False)
    parser.add_argument("--num_gpus_decode", type=int, default=1)
    parser.add_argument("--output_dir", default="outputs/")
    
    return parser

# python train_gsm8k.py --batch_size 3 --save_dir ckpts/debug --train_datapath data/gsm8k/train_ranking_small_full.json --experiment_name gsm8k_ranking_small_full_low_lr