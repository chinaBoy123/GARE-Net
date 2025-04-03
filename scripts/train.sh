export CUDA_VISIBLE_DEVICES=0
DATA_PATH="/home/ubuntu/Students/zhoutao/data"
VOCAB_PATH="/home/ubuntu/Students/zhoutao/data/vocab"
SPLIT=f30k_gru
## f30k
if [ $SPLIT = f30k_gru ];then
  DATASET_NAME="f30k_precomp"
  LOGGER_NAME='/home/ubuntu/Students/zhoutao/code_updated/PFGR/runs/runX/log'
  MODEL_NAME='/home/ubuntu/Students/zhoutao/code_updated/PFGR/runs/runX/checkpoint/f30k_gru'
  nohup /home/ubuntu/miniconda3/envs/zhoutao/bin/python /home/ubuntu/Students/zhoutao/code_updated/PFGR/train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bigru \
  --vocab_path=${VOCAB_PATH} --logger_name=${LOGGER_NAME}/log --model_name=${MODEL_NAME} \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --precomp_enc_type=selfattention --workers=16 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=1 --batch_size=384 \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=LSEPooling --belta=0.1 \
  --drop --wemb_type=glove \
  --criterion=ContrastiveLoss --margin=0.05 \
  > /home/ubuntu/Students/zhoutao/code_updated/PFGR/nohup.out &
fi
if [ $SPLIT = f30k_bert ];then
  DATASET_NAME="f30k_precomp"
  LOGGER_NAME=/home/ubuntu/Students/zhoutao/code_updated/PFGR/runs/runX/log
  MODEL_NAME=/home/ubuntu/Students/zhoutao/code_updated/PFGR/runs/runX/checkpoint/f30k_bert
  nohup /home/ubuntu/miniconda3/envs/zhoutao/bin/python /home/ubuntu/Students/zhoutao/code_updated/PFGR/train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bert \
  --vocab_path=${VOCAB_PATH} --logger_name=${LOGGER_NAME}/log --model_name=${MODEL_NAME} \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --workers=16 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=1 --batch_size=128 \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=MeanPooling --belta=0.1 \
  --drop \
  --criterion=ContrastiveLoss --margin=0.05 \
  > /home/ubuntu/Students/zhoutao/code_updated/PFGR/nohup.out & 
fi
if [ $SPLIT = coco_gru ];then
  DATASET_NAME='coco_precomp'
  LOGGER_NAME=/home/ubuntu/Students/zhoutao/code_updated/PFGR/runs/runX/log
  MODEL_NAME=/home/ubuntu/Students/zhoutao/code_updated/PFGR/runs/runX/checkpoint/coco_gru
  nohup /home/ubuntu/miniconda3/envs/zhoutao/bin/python /home/ubuntu/Students/zhoutao/code_updated/PFGR/train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bigru \
  --vocab_path=${VOCAB_PATH} --logger_name=${LOGGER_NAME}/log --model_name=${MODEL_NAME} \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --precomp_enc_type=selfattention --workers=16 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=1 --batch_size=384 \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=LSEPooling --belta=0.1 \
  --drop --wemb_type=glove \
  --criterion=ContrastiveLoss --margin=0.05 \
  > /home/ubuntu/Students/zhoutao/code_updated/PFGR/nohup.out & 
fi
if [ $SPLIT = coco_bert ];then
  DATASET_NAME='coco_precomp'
  LOGGER_NAME=/home/ubuntu/Students/zhoutao/code_updated/PFGR/runs/runX/log
  MODEL_NAME=/home/ubuntu/Students/zhoutao/code_updated/PFGR/runs/runX/checkpoint/coco_bert
  nohup /home/ubuntu/miniconda3/envs/zhoutao/bin/python /home/ubuntu/Students/zhoutao/code_updated/PFGR/train.py \
  --data_path=${DATA_PATH} --data_name=${DATASET_NAME} --text_enc_type=bert \
  --vocab_path=${VOCAB_PATH} --logger_name=${LOGGER_NAME}/log --model_name=${MODEL_NAME} \
  --num_epochs=25 --lr_update=15 --learning_rate=5e-4 --workers=16 \
  --log_step=200 --embed_size=1024 --vse_mean_warmup_epochs=1 --batch_size=128 \
  --coding_type=VHACoding --alpha=0.1 --pooling_type=MeanPooling --belta=0.1 \
  --drop \
  --criterion=ContrastiveLoss --margin=0.05 \
  > /home/ubuntu/Students/zhoutao/code_updated/PFGR/nohup.out & 
fi