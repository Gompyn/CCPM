TOKENIZERS_PARALLELISM=true python main.py \
          --model_name_or_path results \
          --do_predict --predict_file data/test_public.jsonl \
          --fp16 \
          --per_device_eval_batch_size=64 \
          --dataloader_num_workers=8 \
          --output_dir temp_predict \
          --save_prediction_path temp_predict/CCPM.jsonl
