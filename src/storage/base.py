import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Optional, Iterable
from src.utils.imaging.save_image import save_multiple_cv2_images
from src.utils.imaging.read_image import read_cv2_image
from src.utils.logs import setup_logger


logger = setup_logger("Storage", f"{__name__}.log")


class Storage(ABC):
    """
    An abstract base class defining an interface for storing and retrieving generic objects.
    """
    @abstractmethod
    def write_multiple(self, key: str, values: Iterable):
        """
        Write multiple observations to storage.
        
        Args:
            key: A unique string identifier for the observations
            value: An iterable of data to store (can be any type)
            
        Returns:
            bool: True if write was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def read(self, key: str) -> Optional[Any]:
        """
        Read an observation from storage.
        
        Args:
            key: The string identifier of the observation to retrieve
            
        Returns:
            The stored value if found, None if not found
        """
        pass

# Example implementation (optional) to demonstrate usage:
class ImageStorage(Storage):
    """A simple file-based storage implementation."""
    def write_multiple(self, key: str, values: Iterable[np.array]) -> bool:
        try:
            all_sucess = save_multiple_cv2_images(
                values,
                base_name=key,
                extension=".png"
            )
            return all_sucess
        except Exception:
            return False
    
    def read(self, key: str, color_mode=cv2.IMREAD_COLOR) -> Optional[Any]:
        return read_cv2_image(key, color_mode=color_mode)
