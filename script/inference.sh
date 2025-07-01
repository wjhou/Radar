#!/bin/bash

nvidia-smi
OUTPUT_DIR=./checkpoints/
OUTPUT_DIR_PT="Salesforce/xgen-mm-phi3-mini-instruct-interleave-r-v1.5"
IMAGE_FOLDER="./data/mimic_cxr/images/"
ANNOTATRION_PATH="./data/mimic_cxr/annotation.json"
CLINICAL_CONTEXT_PATH="./data/mimic_cxr/clinical_context.json"
KNOWLEDGE_PATH="./data/mimic_cxr/knowledge_[placeholder].jsonl"
IMAGE_ANNOTATION_PATH="./data/mimic_cxr/expert_annotation.json"
GT_OBSERVATION_PATH="./data/mimic_cxr/observation.json"
# semi-structured knowledge
SENTENCE_KNOWLEDGE_PATH="./data/mimic_cxr/sentence_observation.json"
DEBUG_MODEL=False
BEST_METRIC=eval_BLEU_4
REPORT_TO=none
PER_DEVICE_BATCH_SIZE=2
NUM_TRAIN_EPOCHS=3
ACC_STEPS=4
LR=1e-4
WEIGHT_DECAY=0.0
NUM_BEAMS=1
DATE=$2
STAGE=$3
EVAL_STRAGERY="epoch"
SAVE_STRATEGY="epoch"
SAVE_STEPS=0.25
LOAD_BEST=True
OUTPUT_DIR_FT=${OUTPUT_DIR}/radar_stage1
SFT_DIR=${OUTPUT_DIR}/radar_stage1/

if [ "$1" -ne 1 ];
then
    echo "********** debug **********"
    echo "********** debug **********"
    echo "********** debug **********"
    DEBUG_MODEL=True
    NUM_TRAIN_EPOCHS=1
    REPORT_TO="none"
    OUTPUT_DIR_FT=${OUTPUT_DIR}/debug
fi

localhost="localhost:0,1,2,3"
master_port=29512
per_device_eval_batch_size=20

deepspeed --include ${localhost} --master_port ${master_port} src/inference.py \
    --deepspeed zero2.json \
    --stage $STAGE \
    --model_name_or_path ${OUTPUT_DIR_PT} \
    --sft_model_name_or_path ${SFT_DIR} \
    --data_path ${ANNOTATRION_PATH} \
    --clinical_context_path ${CLINICAL_CONTEXT_PATH} \
    --knowledge_path ${KNOWLEDGE_PATH} \
    --image_annotation_path ${IMAGE_ANNOTATION_PATH} \
    --sentence_knowledge_path ${SENTENCE_KNOWLEDGE_PATH} \
    --gt_observation_path ${GT_OBSERVATION_PATH} \
    --image_folder ${IMAGE_FOLDER} \
    --output_dir ${OUTPUT_DIR_FT} \
    --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
    --gradient_accumulation_steps ${ACC_STEPS} \
    --per_device_eval_batch_size $per_device_eval_batch_size \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --eval_strategy ${EVAL_STRAGERY} \
    --save_strategy ${SAVE_STRATEGY} \
    --save_steps ${SAVE_STEPS} \
    --eval_steps ${SAVE_STEPS} \
    --save_total_limit 1 \
    --learning_rate ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --log_level info \
    --model_max_length 2048 \
    --num_beams ${NUM_BEAMS} \
    --load_best_model_at_end ${LOAD_BEST} \
    --save_safetensors False \
    --dataloader_num_workers 4 \
    --predict_with_generate True \
    --prediction_loss_only False \
    --lazy_preprocess True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters True \
    --debug_model ${DEBUG_MODEL} \
    --metric_for_best_model ${BEST_METRIC} \
    --report_to ${REPORT_TO} \
    --bf16 True