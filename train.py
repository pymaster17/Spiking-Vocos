from pytorch_lightning.cli import LightningCLI


if __name__ == "__main__":
    cli = LightningCLI(run=False, save_config_overwrite=True)
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)
