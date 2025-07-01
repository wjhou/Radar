MODEL_PATH="./checkpoints/expert_model.safetensors"
IMAGE_PATH="./data/chexpert_plus/images/"
ANNOTATION_PATH="./data/chexpert_plus/annotation.json"
OUTPUT_PATH="data/chexpert_plus/knowledge_[placeholder].jsonl"
CLINICAL_CONTEXT_PATH="./data/chexpert_plus/clinical_context.json"
GT_OBSERVATION_PATH="./data/mimic_cxr/observation.json"

./retrieval/retrieval.py \
    --model_path $MODEL_PATH \
    --image_path $IMAGE_PATH \
    --annotation_path $ANNOTATION_PATH \
    --output_path $OUTPUT_PATH \
    --clinical_context_path $CLINICAL_CONTEXT_PATH \
    --gt_observation_path $GT_OBSERVATION_PATH