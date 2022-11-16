import hydra
from omegaconf import DictConfig, OmegaConf
import os
import logging

from Trainer import Trainer

from tools import Log
import sys


LOG = logging.getLogger(__name__)




@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig) -> None:
	log = Log(LOG)

	for cv in ['cv2', 'cv3', 'cv4', 'cv5', 'cv1']:
		cfg.dataset.cv = cv
		log.info("Config:", OmegaConf.to_yaml(cfg))
		log.info("Working directory:", os.getcwd())
		log.debug("Debug level message", None)

		log.start("Trainer initialization")
		trainer = Trainer(cfg, log)
		log.end("Trainer initialization")

		if not cfg.training.only_val:
			log.start("Training")
			trainer.run_training()
			log.end("Training")

		log.start("Eval")
		trainer.run_eval()
		log.end("Eval")




if __name__ == '__main__':
	main()