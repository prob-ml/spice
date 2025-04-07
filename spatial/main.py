import hydra
from hydra import compose, initialize

from omegaconf import OmegaConf


def conditional_resolver(condition, true_val, false_val):
    return true_val if condition else false_val


OmegaConf.register_new_resolver("conditional", conditional_resolver)


@hydra.main(config_path="../config", config_name="configXenium")
def main(cfg):
    if cfg.mode == "train":
        from spatial.train_xenium import train as task
    elif cfg.mode == "generate":
        from spatial.generate_graph import generate_graph as task
    elif cfg.mode == "predict":
        from spatial.predict import test as task
    else:
        raise KeyError
    # print(cfg)
    task(cfg)


if __name__ == "__main__":
    with initialize(config_path="../config"):
        cfg_from_terminal = compose(config_name="configXenium")
        main(cfg_from_terminal)
