MODEL_PATH="./checkpoints/expert_model.safetensors"
IMAGE_PATH="./data/mimic_cxr/images/"
ANNOTATION_PATH="./data/mimic_cxr/annotation.json"
OUTPUT_PATH="data/mimic_cxr/knowledge_[placeholder].jsonl"
CLINICAL_CONTEXT_PATH="./data/mimic_cxr/clinical_context.json"
GT_OBSERVATION_PATH="./data/mimic_cxr/observation.json"

./annotate_retrieve/retrieve.py \
    --model_path $MODEL_PATH \
    --image_path $IMAGE_PATH \
    --annotation_path $ANNOTATION_PATH \
    --output_path $OUTPUT_PATH \
    --clinical_context_path $CLINICAL_CONTEXT_PATH \
    --gt_observation_path $GT_OBSERVATION_PATH