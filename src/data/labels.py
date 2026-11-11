"""Classes partagées par l'entraînement, l'évaluation et l'analyseur."""

CLASSES_3 = ["Nothing", "Check", "Checkmate"]
CLASSES_5 = ["Nothing", "Check White", "Check Black", "Checkmate White", "Checkmate Black"]

MAPPING_5_TO_3 = {
    "Nothing": "Nothing",
    "Check": "Check",
    "Check White": "Check",
    "Check Black": "Check",
    "Checkmate": "Checkmate",
    "Checkmate White": "Checkmate",
    "Checkmate Black": "Checkmate",
}


def get_model_classes(model):
    """Lire les classes sauvegardées, ou les déduire du nombre de sorties."""
    if model.classes:
        return model.classes
    return CLASSES_5 if model.output_dim == 5 else CLASSES_3
