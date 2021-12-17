TOKENIZERS_PARALLELISM=true python main.py \
          --model_name_or_path hfl/chinese-roberta-wwm-ext \
          --do_train --train_file data/train.jsonl \
          --do_eval  --validation_file data/valid.jsonl \
          --learning_rate 5e-5  --fp16 \
          --num_train_epochs 10 \
          --output_dir results \
          --per_device_eval_batch_size=64 \
          --per_device_train_batch_size=64 \
          --overwrite_output \
          --evaluation_strategy 'epoch' \
          --save_strategy 'epoch' \
          --dataloader_num_workers=8 \
          --load_best_model_at_end \
          --metric_for_best_model 'accuracy' \
          --greater_is_better True 

# hfl/chinese-roberta-wwm-ext
# SIKU-BERT/sikubert