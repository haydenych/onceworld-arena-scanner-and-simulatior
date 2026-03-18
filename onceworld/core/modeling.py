"""Shared model construction helpers for training and inference."""


def build_resnet18_classifier(num_classes, models_module, nn_module):
    model = models_module.resnet18(weights=None)
    model.fc = nn_module.Linear(model.fc.in_features, int(num_classes))
    return model

