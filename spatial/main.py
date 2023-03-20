import hydra
from hydra import compose, initialize

# import torch
# from torch.profiler import profile, record_function, ProfilerActivity


# @hydra.main(config_path="../config", config_name="config")
# def main(cfg):
#     with profile(
#         activities=[ProfilerActivity.CUDA],
#         record_shapes=True,
#         profile_memory=True,
#         on_trace_ready=torch.profiler.tensorboard_trace_handler(
#             cfg.paths.output + "/" + cfg.training.logger_name
#         ),
#     ):
#         with record_function("model_inference"):
#             if cfg.mode == "train":
#                 from spatial.train import train as task
#             elif cfg.mode == "generate":
#                 from spatial.generate_graph import generate_graph as task
#             elif cfg.mode == "predict":
#                 from spatial.predict import test as task
#             else:
#                 raise KeyError
#             print(cfg)
#             task(cfg)


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
    # print(cfg)
    task(cfg)


if __name__ == "__main__":
    with initialize(config_path="../config"):
        cfg_from_terminal = compose(config_name="config")
        main(cfg_from_terminal)
