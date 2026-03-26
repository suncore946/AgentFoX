import json
import numpy as np


class CustomJsonEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            if np.isnan(obj):
                return None
            return round(float(obj), 6)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        # 处理 np.nan 值
        elif obj is np.nan or (hasattr(np, "isnan") and np.isnan(obj)) or obj is None:
            return None
        return super().default(obj)
