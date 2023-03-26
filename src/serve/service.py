import torch
import bentoml
from bentoml.io import Image, JSON
from torchvision import transforms
from config import LABELS

torchscript_runner = bentoml.torchscript.get(
    "torchscript-attribute-classifier"
).to_runner()

svc = bentoml.Service(
    "face-attribute-classifier",
    runners=[torchscript_runner],
)


process_image = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5063, 0.4258, 0.3832], std=[0.2644, 0.2436, 0.2397]
        ),
    ]
)


@svc.api(input=Image(), output=JSON())
def classify(input_image):
    tensor = process_image(input_image)
    tensor = torch.unsqueeze(tensor, 0)
    print(f"TENSOR SHAPE : {tensor.shape}")

    predictions = torchscript_runner.run(tensor)
    _, indices = torch.where(predictions > 0.3)
    indices = indices.numpy()
    output = {}
    labels = []
    for index in indices:
        probability = round(float(predictions[0][index]), 3)
        labels.append(
            {
                "label_id": index,
                "label": LABELS[index],
                "probability": probability,
            }
        )
    output["labels"] = labels

    print(output)

    return output
