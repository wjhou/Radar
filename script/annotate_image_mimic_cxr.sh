MODEL_PATH="./checkpoints/expert_model.safetensors"
IMAGE_PATH="./data/mimic_cxr/images/"
ANNOTATION_PATH="./data/mimic_cxr/annotation.json"
OUTPUT_PATH="data/mimic_cxr/expert_annotation.json"
CLINICAL_CONTEXT_PATH="./data/mimic_cxr/clinical_context.json"

./annotate_retrieve/annotate_image.py \
    --model_path $MODEL_PATH \
    --image_path $IMAGE_PATH \
    --annotation_path $ANNOTATION_PATH \
    --clinical_context_path $CLINICAL_CONTEXT_PATH \
    --output_path $OUTPUT_PATH