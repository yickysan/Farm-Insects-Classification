from FarmInsectsClassifier.torch.config.configuration import CallBackConfigManager
from FarmInsectsClassifier.torch.components.callbacks import get_call_backs, EarlyStopper, TensorBoardWriter, SaveBestModel
from FarmInsectsClassifier.exception import CallBackError
from FarmInsectsClassifier.logger import logging


class TorchCallBacksPipeline:

    def __init__(self) -> None:
        self.stage_name = "Callbacks Preparation"
    
    def initiate_pipeline(self, **kwargs) -> tuple[EarlyStopper, TensorBoardWriter, SaveBestModel]:
        """
        Parameters
        ----------
        kwargs[patience] : int
        kwargs[min_delta] : float
        
        Returns
        -------
        Tuple 
            Tuple of EarlyStopper, TensorBoardWriter, SaveBestModel
        """

        patience = kwargs["patience"]
        min_delta = kwargs["min_delta"]
        
        try:
            callback_config_manager = CallBackConfigManager()
            callback_config = callback_config_manager.get_config()
            callbacks = get_call_backs(callback_config, patience, min_delta)

            return callbacks
        
        except Exception as e:
            logging.info(e)
            raise CallBackError(e)