import argparse
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported
from finetune.lib import prepare_peft_model_n_tokenizer, \
        prepare_dataset, show_memory_stat, max_seq_length, \
        seed_everything 
from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only
import pdb 

def main(args):
    seed = 3407
    seed_everything(seed)
    
    model, tokenizer = prepare_peft_model_n_tokenizer(
        model_name=args.model
        #model_name='mistralai/Mamba-Codestral-7B-v0.1'
    )
    # train_dataset, eval_dataset  = prepare_dataset(tokenizer=tokenizer)
    train_dataset, eval_dataset  = prepare_dataset(tokenizer=tokenizer, dataset_dir=args.dataset_dir)
    
    # if args.max_steps != -1:
    #     args.epoch = 1

    trainer_args = TrainingArguments(
            per_device_train_batch_size = args.bs,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            num_train_epochs = args.epoch, # Set this for 1 full training run.
            # max_steps = args.max_steps,
            learning_rate = 2e-4,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = seed,
            output_dir = "outputs",
            report_to = "wandb", # Use this for WandB etc
            # add evaluation arguments
            fp16_full_eval = True,
            per_device_eval_batch_size = 2,
            eval_accumulation_steps = 4,
            eval_strategy = "epoch",
            eval_steps = 1,
        )
    
    trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    args=trainer_args,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    show_memory_stat()
    min_eval_loss = 1e5
    for r in range(args.max_round):
        print(f'Running round #{r}..')
        trainer_stats = trainer.train()
        eval_stats = trainer.evaluate()
        if eval_stats['eval_loss'] < min_eval_loss:
            min_eval_loss = eval_stats['eval_loss']
        
            # Save model and tokenizer
            model.save_pretrained(args.model.split('/')[-1]) # Local saving
            tokenizer.save_pretrained(args.model.split('/')[-1])
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # default it to use gpt-4o
    parser.add_argument(
        "--model", type=str, default='unsloth/Llama-3.2-1B', 
        help="default use Llama-3.2-3B-Instruct"
    )
    parser.add_argument(
        "--dataset_dir", type=str, default="/home/nesl/oliver/timeSeriesMamba/LLMFinetune/jsons/combo1.json", 
        help="Your finetuning json file"
    )
    parser.add_argument(
        "--bs", type=int, default=2, 
        help="batch size"
    )
    parser.add_argument(
        "--epoch", type=float, default=2, 
        help="epoch"
    )
    parser.add_argument(
        "--max_round", type=int, default=1, 
        help="max_round"
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, 
        help="maximum finetuning step"
    )
    args = parser.parse_args()
    main(args)