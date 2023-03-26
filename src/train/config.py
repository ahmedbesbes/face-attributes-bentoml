from dataclasses import dataclass


NUM_LABELS = 40


@dataclass
class TrainingConfig:
    n_epochs: int
    n_samples: int
    batch_size: int
    image_folder_path: str
    csv_data_path: str
    test_size: float
    lr: float
    num_workers: int
