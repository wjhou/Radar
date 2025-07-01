ANNOTATION_PATH="./checkpoints/radar_stage1/"
CHEXBERT_PATH="./checkpoints/chexbert.pth"
OUTPUT_OBSERVATION_PATH="./checkpoints/radar_stage1/observation.json"
OUTPUT_SENTENCE_OBSERVATION_PATH="./checkpoints/radar_stage1/sentence_observation.json"

./annotate_retrieve/annotate_hypothesis.py \
    --annotation_path $ANNOTATION_PATH \
    --chexbert_path $CHEXBERT_PATH \
    --output_observation_path $OUTPUT_OBSERVATION_PATH \
    --output_sentence_observation_path $OUTPUT_SENTENCE_OBSERVATION_PATH