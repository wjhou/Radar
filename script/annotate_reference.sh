ANNOTATION_PATH="./data/mimic_cxr/annotation.json"
CHEXBERT_PATH="./checkpoints/chexbert.pth"
OUTPUT_OBSERVATION_PATH="./data/mimic_cxr/observation.json"
OUTPUT_SENTENCE_OBSERVATION_PATH="./data/mimic_cxr/sentence_observation.json"

./annotate_retrieve/annotate_reference.py \
    --annotation_path $ANNOTATION_PATH \
    --chexbert_path $CHEXBERT_PATH \
    --output_observation_path $OUTPUT_OBSERVATION_PATH \
    --output_sentence_observation_path $OUTPUT_SENTENCE_OBSERVATION_PATH