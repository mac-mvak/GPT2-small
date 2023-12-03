import wandb


class Writer:
    def __init__(self, project, name, cfg) -> None:
        wandb.init(project=project, name=name, 
                   config=cfg)
    
    def log(self, msg):
        wandb.log(msg)

    def finish(self):
        wandb.finish()

