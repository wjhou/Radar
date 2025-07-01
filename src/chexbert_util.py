import torch
from collections import OrderedDict


CONDITIONS = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]
CONDITIONS_PRINT = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Enlarged Cardiomediastinum",
    "Lung Opacity",
    "Lung Lesion",
    "Pneumonia",
    "Pneumothorax",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]
TOP5_CONDITIONS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]


def load_chexbert(checkpoint_path):
    import sys

    sys.path.append("../CheXbert/src/")
    from models.bert_labeler import bert_labeler

    chexbert = bert_labeler()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in checkpoint["model_state_dict"].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    chexbert.load_state_dict(new_state_dict)
    print("Loaded reward model from {}".format(checkpoint_path))
    chexbert.eval()
    return chexbert.cuda()
