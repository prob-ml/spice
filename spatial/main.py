import hydra


@hydra.main(config_path="../config", config_name="config")
def main(cfg):
    if cfg.mode == "train":
        from spatial.train import train as task
    elif cfg.mode == "generate":
        from spatial.generate_graph import generate_graph as task
    elif cfg.mode == "predict":
        from spatial.predict import test as task
    else:
        raise KeyError
    task(cfg)
