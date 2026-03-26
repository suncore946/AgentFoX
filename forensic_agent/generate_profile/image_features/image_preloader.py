import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from typing import Dict, List

from ...utils.image_utils import load_image


class ImagePreloader:
    """图像预加载器"""

    def __init__(self, n_threads: int = 8):
        self.n_threads = n_threads
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=self.n_threads)  # 在实例化时创建线程池

    def load_images_batch(self, image_paths: List[str]) -> Dict[str, np.ndarray]:
        """批量预加载图像"""
        loaded_images = {}

        def load_single_image(path: str) -> tuple:
            image = load_image(path, return_pil=True)
            return path, image

        futures = {self.executor.submit(load_single_image, path): path for path in image_paths}

        for future in as_completed(futures):
            path, image = future.result()
            if image is not None:
                loaded_images[path] = image

        return loaded_images

    def __del__(self):
        self.executor.shutdown(wait=True)  # 确保线程池在对象销毁时正确关闭
