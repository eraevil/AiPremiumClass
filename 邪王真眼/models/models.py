import torch
import torch.nn as nn

from enum import Enum
from typing import Any, Dict


class ModelType(Enum):
    Logistic_Regression = 'logistic_regression'
    ResNet = 'resnet'
    RNN = 'rnn'
    CBOW = 'cbow'
    SkipGram = 'skip_gram'
    CNN = 'cnn'
    BertClassifier = 'bert_classifier'
    SIMPLE_ATTENTION = 'simple_attention'

    @classmethod
    def from_str(cls, label: str) -> "ModelType":
        if label in cls.__members__:
            return cls[label]
        
        for member in cls:
            if member.value.lower() == label.lower():
                return member
        raise ValueError(f"Unknown ModelType: {label!r}. "
                         f"Valid names: {list(cls.__members__.keys())}, "
                         f"values: {[m.value for m in cls]}")


def get_model(cfg: Dict[str, Any]):
    model_type = cfg.get('model')
    if isinstance(model_type, str):
        model_type = ModelType.from_str(model_type)
    model = None

    if model_type == ModelType.Logistic_Regression:
        from .logistic_regression import LogisticRegression
        model = LogisticRegression(cfg)
    elif model_type == ModelType.ResNet:
        from .resnet import ResNet
        model = ResNet(cfg)
    elif model_type == ModelType.RNN:
        from .rnn import RNN
        model = RNN(cfg)
    elif model_type == ModelType.CBOW:
        from .cbow import CBOW
        model = CBOW(cfg)
    elif model_type == ModelType.SkipGram:
        from .skip_gram import SkipGram
        model = SkipGram(cfg)
    elif model_type == ModelType.CNN:
        from .cnn import CNN
        model = CNN(cfg)
    elif model_type == ModelType.BertClassifier:
        from .bert_classifier import BertClassifier
        model = BertClassifier(cfg)
    elif model_type == ModelType.SIMPLE_ATTENTION:
        from .simple_attention import SimpleAttention
        model = SimpleAttention(cfg)
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    return model
