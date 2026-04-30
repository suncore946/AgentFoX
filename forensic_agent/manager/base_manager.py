"""Base manager interface.

中文说明: Manager 类用于封装配置、数据、图片、日志等运行时服务。
English: Manager classes wrap runtime services such as configuration, data,
images, and logging.
"""

from abc import ABC, abstractmethod


class BaseManager(ABC):
    """Minimal manager base class.

    中文说明: run 是可选扩展接口, 默认不做任何操作。
    English: run is an optional extension point and does nothing by default.
    """

    def run(self, *args, **kwargs):
        """Optional execution hook.

        中文说明: 当前最小推理链路不依赖该方法。
        English: The minimal inference path does not depend on this method.
        """
        pass
