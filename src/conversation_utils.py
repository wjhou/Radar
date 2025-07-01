IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = "<image>"
SYSTEM_PROMPT = "You are an assistant in radiology, responsible for analyzing medical imaging studies and generating detailed, structured, and accurate radiology reports."
INSTRUCTION_PROMPT = "Generate a comprehensive and detailed description of findings based on this chest X-ray image."
INSTRUCTION_PRIOR_PROMPT = "Generate a comprehensive and detailed description of findings based on this chest X-ray image. Include a thorough comparison with a prior chest X-ray, emphasizing any significant changes, progression, or improvement."
OBSERVATION_PROMPT = "Before this, systematically identify all observations."


def preprocess_phi_3_new(sources, tokenizer):
    role_mapping = {"human": "user", "gpt": "assistant"}
    roles = ("<|user|>", "<|assistant|>")
    sep = "<s>"
    sep2 = "<|end|>"

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        for i, rnd in enumerate(source):
            if "from" in rnd:
                if rnd["from"] in ["human", "gpt"]:
                    rnd["role"] = role_mapping[rnd.pop("from")]
                else:
                    rnd["role"] = rnd.pop("from")
            if "value" in rnd:
                rnd["content"] = rnd.pop("value")
        chat_conv = tokenizer.apply_chat_template(source, tokenize=False)
        chat_conv = chat_conv.replace(tokenizer.bos_token, "")
        conversations.append(chat_conv)

    # Tokenize conversations
    if tokenizer.model_max_length > 2048:
        max_len = 2048
    else:
        max_len = tokenizer.model_max_length

    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=max_len,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # Mask targets
    sep = roles[1] + "\n"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())
        rounds = conversation.split(sep2 + "\n")
        cur_len = 0
        for i, rou in enumerate(rounds):
            if rou == "":
                break
            rou += sep2 + "\n"
            if sep in rou:
                # assistant round
                round_ids = tokenizer(
                    rou, max_length=max_len, truncation=True
                ).input_ids
                role_prefix_ids = tokenizer(sep).input_ids
                len_prefix = len(role_prefix_ids)
                round_ids = round_ids[len_prefix:]
                round_len = len(round_ids)
            elif roles[0] in rou:
                # user round
                rou += sep
                round_ids = tokenizer(
                    rou, max_length=max_len, truncation=True
                ).input_ids
                round_len = len(round_ids)
                instruction_len = round_len  # All are instructions.
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            else:
                round_ids = tokenizer(
                    rou, max_length=max_len, truncation=True
                ).input_ids
                round_len = len(round_ids)
                instruction_len = round_len  # All are instructions.
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX
            cur_len += round_len

    return dict(
        input_ids=input_ids,
        labels=targets,
    )
