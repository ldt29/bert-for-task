# Bert

## Pre-trained model
I have cloned  huawei-noah/TinyBERT_General_4L_312D to local which is ../TinyBERT_General_4L_312D
```
git lfs install
git clone https://huggingface.co/huawei-noah/TinyBERT_General_4L_312D
```
or  
```
git lfs install
git clone git@hf.co:huawei-noah/TinyBERT_General_4L_312D
```
## GPU

Train batch size should be 32 or 64. So if i use 8 GPUs, train batch size per gpu should be 4 or 8.

## BertForSequenceClassification
### train

default command:  
```
python classifier.py  --model_type bert  --model_name_or_path ../TinyBERT_General_4L_312D  --task_name yelp  --do_train  --do_eval  --data_dir ../yelp_small  --per_gpu_train_batch_size 8  --learning_rate 3e-5  --num_train_epochs 3.0  --logging_steps 50  --max_seq_length 128  --output_dir ../bert-for-cls-default  --evaluate_during_training  --logging_steps 50  --overwrite_output_dir  --overwrite_cache 
```

best model command:    
```
python classifier.py  --model_type bert  --model_name_or_path ../TinyBERT_General_4L_312D  --task_name yelp  --do_train  --do_eval  --data_dir ../yelp_small  --per_gpu_train_batch_size 4 --learning_rate 3e-5  --num_train_epochs 10.0  --logging_steps 50  --max_seq_length 128  --output_dir ../bert-for-cls-finetune  --evaluate_during_training  --logging_steps 50  --overwrite_output_dir  --overwrite_cache
```
### test
default command:  
```
python classifier.py  --model_type bert  --model_name_or_path ../TinyBERT_General_4L_312D  --task_name yelp  --do_eval  --data_dir ../yelp_small  --output_dir ../bert-for-cls-default
```
dev acc: 0.486  
test acc: 0.554  

best model command:  
```
python classifier.py  --model_type bert  --model_name_or_path ../TinyBERT_General_4L_312D  --task_name yelp  --do_eval  --data_dir ../yelp_small  --output_dir ../bert-for-cls-finetune
```
dev acc: 0.518  
test acc: 0.57 
  

## BertForMaskedLanguageModel
### train

best model command:    
```
python classifier_mlm.py  --model_type bert  --model_name_or_path ../TinyBERT_General_4L_312D  --task_name yelp  --do_train  --do_eval  --data_dir ../yelp_small  --per_gpu_train_batch_size 4  --learning_rate 3e-5  --num_train_epochs 10.0  --logging_steps 50  --max_seq_length 128  --mlm_alpha 1.0  --output_dir ../bert-for-clsmlm-finetune  --evaluate_during_training  --train_data_file ../yelp_small/unlabeled_train_50000.txt  --block_size 128  --line_by_line  --overwrite_output_dir  --overwrite_cache
```

labeled training data to 50:  
Use small batch size to train (add --small_labeled_data): 
```
python classifier_mlm.py  --model_type bert  --model_name_or_path ../TinyBERT_General_4L_312D  --task_name yelp  --do_train  --do_eval  --data_dir ../yelp_small  --per_gpu_train_batch_size 4  --learning_rate 3e-5  --num_train_epochs 1000.0  --logging_steps 50  --max_seq_length 128  --mlm_alpha 1.0  --output_dir ../bert-for-50labeled-finetune  --evaluate_during_training  --train_data_file ../yelp_small/unlabeled_train_50000.txt  --block_size 128  --line_by_line  --overwrite_output_dir  --overwrite_cache --small_labeled_data 
```
### test
best model command: 
```
python classifier_mlm.py  --model_type bert  --model_name_or_path ../TinyBERT_General_4L_312D  --task_name yelp  --do_eval  --data_dir ../yelp_small  --output_dir ../bert-for-clsmlm-finetune
```
dev acc: 0.528  
test acc: 0.576  

labeled training data to 50: 
```
python classifier_mlm.py  --model_type bert  --model_name_or_path ../TinyBERT_General_4L_312D  --task_name yelp  --do_eval  --data_dir ../yelp_small  --output_dir ../bert-for-50labeled-finetune 
```

dev acc: 0.366  
test acc: 0.398   