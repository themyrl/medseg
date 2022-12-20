import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging

from TrainerV2 import Trainer

from tools import Log
import sys


LOG = logging.getLogger(__name__)




@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
	log = Log(LOG)

	# if len(sys.argv) > 2:
	# 	if sys.argv[1] == '-c':
	# 		cfg = OmegaConf.load(sys.argv[2])

	log.info("Config:", OmegaConf.to_yaml(cfg))
	log.info("Working directory:", os.getcwd())
	log.debug("Debug level message", None)

	log.start("Trainer initialization")
	trainer = Trainer(cfg, log)
	log.end("Trainer initialization")
	# print(cfg.training.only_val)
	# exit(0)
	if not cfg.training.only_val:
		log.start("Training")
		trainer.run_training()
		log.end("Training")

	log.start("Eval")
	trainer.run_eval()
	log.end("Eval")




if __name__ == '__main__':
	main()