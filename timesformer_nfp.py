import torch
import torch.nn as nn
import transformers
from transformers import AutoModelForNextFramePrediction
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from fvd import calculate_fvd
from is import calculate_is

class TimesFormerForNextFramePrediction(LightningModule):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.model = AutoModelForNextFramePrediction.from_pretrained(config.model_name)

  def forward(self, x):
    return self.model(x)

  def training_step(self, batch):
    loss = self.model(batch["video"]).loss
    self.log("train_loss", loss, prog_bar=True)
    return loss

  def test_step(self, batch):
    loss = self.model(batch["video"]).loss
    self.log("test_loss", loss, prog_bar=True)
    generated_image = self.model.generate(batch["video"])
    save_image(generated_image, "generated_image.jpg")
    fvd_score = calculate_fvd(batch["video"], generated_image)
    is_score = calculate_is(batch["video"], generated_image)
    self.log("fvd_score", fvd_score, prog_bar=True)
    self.log("is_score", is_score, prog_bar=True)
    return loss

def main():
  config = transformers.TimeSFormerConfig()
  model = TimesFormerForNextFramePrediction(config)
  test_dataset = VirtualKITTIDataset(config.data_path, "val", config.patch_size, config.num_frames)
  test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
  trainer = pl.Trainer(max_epochs=config.num_epochs, optimizer=optimizer)
  trainer.test(model, test_loader)

if __name__ == "__main__":
  main()
