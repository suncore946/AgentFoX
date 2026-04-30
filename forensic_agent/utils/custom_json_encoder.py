"""JSON encoder for numpy values.

中文说明: 特征提取会产生 numpy 标量和数组, 缓存前需要转换为普通 JSON 类型。
English: Feature extraction produces numpy scalars and arrays, which must be
converted to plain JSON types before caching.
"""

import json
import numpy as np


class CustomJsonEncoder(json.JSONEncoder):
    """Encode numpy values for JSON.

    中文说明: NaN 会转成 None, 避免生成非法 JSON。
    English: NaN values are converted to None to avoid invalid JSON.
    """

    def default(self, obj):
        """Convert supported numpy objects.

        中文说明: 未识别对象交给父类处理。
        English: Unknown objects are delegated to the parent encoder.
        """
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            if np.isnan(obj):
                return None
            return round(float(obj), 6)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is np.nan or (hasattr(np, "isnan") and np.isnan(obj)) or obj is None:
            return None
        return super().default(obj)
