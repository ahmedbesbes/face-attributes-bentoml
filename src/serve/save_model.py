import bentoml
import torch


path = "models/ts_model.pt"
script = torch.jit.load(path)
bentoml.torchscript.save_model(
    "torchscript-attribute-classifier",
    script,
    signatures={"__call__": {"batchable": True}},
)
