import wandb
import os

def get_config_dict(args):
    config_dict = {}
    config_dict["model"] = args.model_id
    config_dict["sequence_length"] = args.sequence_length

    # If ppl model
    if hasattr(args, "dataset") and args.dataset:
        config_dict["dataset"] = args.dataset
    
    # If thresh/ base hf
    if args.use_thresh:
        config_dict["method"] = "thresh"
        config_dict["percentile"] = args.percentile
        config_dict["warmup"] = args.warmup
    else:
        config_dict["method"] = "base_hf"
    
    return config_dict

class WandbLogger:
    def __init__(self, args, groupid):
        self.rank = os.environ.get("RANK", "0")
        if self.rank == '0':
            self.config = get_config_dict(args)

            # Project, name, id
            jobid = os.environ.get("JOBID", '0')
            self.run = wandb.init(project='thresh_attn', config=self.config, name=jobid, 
                                  group=groupid, job_type='eval', tags=[groupid])

            wandb.define_metric("compression_ratio", summary="mean")
    
    def update_config(self, kwargs):
        if self.rank == '0':
            self.run.config.update(kwargs)

    def log(self, kwargs):
      if self.rank == '0':
          self.run.log(kwargs)
    
    def log_tasks(self, results):
      if self.rank == '0':
          assert results is not None
          for task, metadata in results.items():
            for metric, result in metadata.items():  
              if metric == 'alias' or metric == ' ' or not metric:
                continue
              metric = metric.replace(",none", "")
              self.run.log({task + "_" + metric: result})
    
    def finish(self):
      if self.rank == '0':
          self.run.finish()