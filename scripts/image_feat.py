class ImageFeatureExtractor:
    """图像特征提取器，独立的特征提取逻辑"""

    @staticmethod
    def extract_features(image: np.ndarray) -> Dict[str, Any]:
        """提取图像的各种特征数据"""
        try:
            h, w, c = image.shape
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            # 提取各类特征
            basic_info = ImageFeatureExtractor._extract_basic_info(image)
            edge_features = ImageFeatureExtractor._extract_edge_features(gray)
            texture_features = ImageFeatureExtractor._extract_texture_features(gray)
            color_features = ImageFeatureExtractor._extract_color_features(image)
            hsv_features = ImageFeatureExtractor._extract_hsv_features(image)
            geometric_features = ImageFeatureExtractor._extract_geometric_features(gray)
            frequency_features = ImageFeatureExtractor._extract_frequency_features(gray)
            compression_features = ImageFeatureExtractor._extract_compression_features(image)

            return {
                "basic_info": basic_info,
                "edge_features": edge_features,
                "texture_features": texture_features,
                "color_features": color_features,
                "hsv_features": hsv_features,
                "geometric_features": geometric_features,
                "frequency_features": frequency_features,
                "compression_features": compression_features,
            }

        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return {"error": str(e)}

    @staticmethod
    def _extract_basic_info(image: np.ndarray) -> Dict[str, Any]:
        """提取基础图像信息"""
        h, w, c = image.shape
        return {"dimensions": [h, w, c], "file_size_estimate": h * w * c, "aspect_ratio": w / h, "total_pixels": h * w}

    @staticmethod
    def _extract_edge_features(gray: np.ndarray) -> Dict[str, Any]:
        """提取边缘特征"""
        edges = cv2.Canny(gray, 50, 150)
        return {
            "edge_density": float(np.sum(edges > 0) / edges.size),
            "edge_count_estimate": int(np.sum(edges > 0)),
            "max_edge_intensity": int(np.max(edges)),
            "edge_distribution": ImageFeatureExtractor._get_spatial_distribution(edges),
        }

    @staticmethod
    def _extract_texture_features(gray: np.ndarray) -> Dict[str, Any]:
        """提取纹理特征"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return {
            "texture_variance": float(np.var(laplacian)),
            "texture_mean": float(np.mean(np.abs(laplacian))),
            "texture_range": [float(np.min(laplacian)), float(np.max(laplacian))],
            "smoothness_areas": ImageFeatureExtractor._detect_smooth_areas(gray),
        }

    @staticmethod
    def _extract_color_features(image: np.ndarray) -> Dict[str, Any]:
        """提取颜色特征"""
        h, w, c = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return {
            "rgb_means": [float(np.mean(image[:, :, i])) for i in range(3)],
            "rgb_stds": [float(np.std(image[:, :, i])) for i in range(3)],
            "brightness_distribution": ImageFeatureExtractor._get_brightness_histogram(gray),
            "color_complexity": ImageFeatureExtractor._calculate_color_complexity(image),
        }

    @staticmethod
    def _extract_hsv_features(image: np.ndarray) -> Dict[str, Any]:
        """提取HSV特征"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        return {
            "hue_distribution": ImageFeatureExtractor._get_hue_distribution(hsv[:, :, 0]),
            "saturation_stats": {
                "mean": float(np.mean(hsv[:, :, 1])),
                "std": float(np.std(hsv[:, :, 1])),
                "range": [float(np.min(hsv[:, :, 1])), float(np.max(hsv[:, :, 1]))],
            },
            "value_stats": {
                "mean": float(np.mean(hsv[:, :, 2])),
                "std": float(np.std(hsv[:, :, 2])),
                "range": [float(np.min(hsv[:, :, 2])), float(np.max(hsv[:, :, 2]))],
            },
        }

    @staticmethod
    def _extract_geometric_features(gray: np.ndarray) -> Dict[str, Any]:
        """提取几何特征"""
        edges = cv2.Canny(gray, 50, 150)
        return {
            "contour_count": ImageFeatureExtractor._count_major_contours(edges),
            "line_segments": ImageFeatureExtractor._detect_line_segments(edges),
            "symmetry_analysis": ImageFeatureExtractor._analyze_symmetry(gray),
            "region_uniformity": ImageFeatureExtractor._analyze_region_uniformity(gray),
        }

    @staticmethod
    def _extract_frequency_features(gray: np.ndarray) -> Dict[str, Any]:
        """提取频域特征"""
        # FFT分析
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.abs(f_shift)

        h, w = magnitude_spectrum.shape
        center_y, center_x = h // 2, w // 2

        # 计算不同频率范围的能量
        low_freq_mask = np.zeros((h, w))
        high_freq_mask = np.zeros((h, w))

        cv2.circle(low_freq_mask, (center_x, center_y), min(h, w) // 8, 1, -1)
        cv2.circle(high_freq_mask, (center_x, center_y), min(h, w) // 4, 1, -1)
        high_freq_mask = high_freq_mask - low_freq_mask

        low_freq_energy = np.sum(magnitude_spectrum * low_freq_mask)
        high_freq_energy = np.sum(magnitude_spectrum * high_freq_mask)
        total_energy = np.sum(magnitude_spectrum)

        return {
            "low_freq_ratio": float(low_freq_energy / total_energy),
            "high_freq_ratio": float(high_freq_energy / total_energy),
            "frequency_concentration": float(low_freq_energy / (low_freq_energy + high_freq_energy)),
            "spectrum_peak_value": float(np.max(magnitude_spectrum)),
            "spectrum_mean": float(np.mean(magnitude_spectrum)),
        }

    @staticmethod
    def _extract_compression_features(image: np.ndarray) -> Dict[str, Any]:
        """提取压缩/噪声特征"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        return {
            "noise_estimate": float(np.std(cv2.GaussianBlur(gray, (5, 5), 0) - gray)),
            "jpeg_artifact_indicators": ImageFeatureExtractor._detect_jpeg_artifacts(gray),
            "compression_estimate": ImageFeatureExtractor._estimate_compression_level(image),
        }

    # 以下是辅助方法的实现（保持不变）
    @staticmethod
    def _get_spatial_distribution(edges: np.ndarray) -> Dict[str, float]:
        """获取边缘的空间分布"""
        h, w = edges.shape
        regions = []
        for i in range(3):
            for j in range(3):
                start_h, end_h = i * h // 3, (i + 1) * h // 3
                start_w, end_w = j * w // 3, (j + 1) * w // 3
                region = edges[start_h:end_h, start_w:end_w]
                regions.append(float(np.sum(region > 0) / region.size))

        return {
            "region_densities": regions,
            "center_density": regions[4],
            "border_avg_density": float(np.mean([regions[0], regions[2], regions[6], regions[8]])),
            "distribution_variance": float(np.var(regions)),
        }

    @staticmethod
    def _detect_smooth_areas(gray: np.ndarray) -> Dict[str, Any]:
        """检测平滑区域"""
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean) ** 2, -1, kernel)

        smooth_threshold = 10
        smooth_areas = local_variance < smooth_threshold

        return {
            "smooth_area_ratio": float(np.sum(smooth_areas) / smooth_areas.size),
            "smooth_regions_count": ImageFeatureExtractor._count_connected_regions(smooth_areas),
            "avg_smoothness": float(np.mean(local_variance)),
        }

    @staticmethod
    def _get_brightness_histogram(gray: np.ndarray) -> List[int]:
        """获取亮度直方图"""
        hist, _ = np.histogram(gray, bins=10, range=(0, 256))
        return hist.tolist()

    @staticmethod
    def _calculate_color_complexity(image: np.ndarray) -> Dict[str, float]:
        """计算颜色复杂度"""
        h, w, c = image.shape
        reduced = image // 32  # 8级量化
        reshaped = reduced.reshape(-1, c)
        unique_colors = len(np.unique(reshaped.view(np.void), axis=0))

        return {
            "unique_colors_estimate": unique_colors,
            "color_diversity_ratio": unique_colors / (h * w),
            "channel_correlations": [
                float(np.corrcoef(image[:, :, 0].flatten(), image[:, :, 1].flatten())[0, 1]),
                float(np.corrcoef(image[:, :, 1].flatten(), image[:, :, 2].flatten())[0, 1]),
                float(np.corrcoef(image[:, :, 0].flatten(), image[:, :, 2].flatten())[0, 1]),
            ],
        }

    @staticmethod
    def _get_hue_distribution(hue: np.ndarray) -> Dict[str, Any]:
        """获取色相分布"""
        hist, _ = np.histogram(hue, bins=18, range=(0, 180))
        dominant_hue_index = np.argmax(hist)

        return {
            "histogram": hist.tolist(),
            "dominant_hue_index": int(dominant_hue_index),
            "dominant_hue_percentage": float(hist[dominant_hue_index] / hue.size),
            "hue_diversity": int(np.sum(hist > hue.size * 0.01)),
        }

    @staticmethod
    def _count_major_contours(edges: np.ndarray) -> int:
        """计算主要轮廓数量"""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        major_contours = [c for c in contours if cv2.contourArea(c) > edges.size * 0.001]
        return len(major_contours)

    @staticmethod
    def _detect_line_segments(edges: np.ndarray) -> Dict[str, Any]:
        """检测线段"""
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        if lines is not None:
            lengths = []
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                angle = np.arctan2(y2 - y1, x2 - x1)
                lengths.append(length)
                angles.append(angle)

            return {
                "line_count": len(lines),
                "avg_length": float(np.mean(lengths)) if lengths else 0,
                "length_variance": float(np.var(lengths)) if lengths else 0,
                "angle_variance": float(np.var(angles)) if angles else 0,
                "has_dominant_direction": float(np.var(angles)) < 0.5 if angles else False,
            }
        else:
            return {"line_count": 0, "avg_length": 0, "length_variance": 0, "angle_variance": 0, "has_dominant_direction": False}

    @staticmethod
    def _analyze_symmetry(gray: np.ndarray) -> Dict[str, float]:
        """分析对称性"""
        h, w = gray.shape

        # 水平对称
        left_half = gray[:, : w // 2]
        right_half = np.fliplr(gray[:, w // 2 : w // 2 + left_half.shape[1]])
        horizontal_symmetry = float(1.0 - np.mean(np.abs(left_half - right_half)) / 255)

        # 垂直对称
        top_half = gray[: h // 2, :]
        bottom_half = np.flipud(gray[h // 2 : h // 2 + top_half.shape[0], :])
        vertical_symmetry = float(1.0 - np.mean(np.abs(top_half - bottom_half)) / 255)

        return {
            "horizontal_symmetry": horizontal_symmetry,
            "vertical_symmetry": vertical_symmetry,
            "overall_symmetry": (horizontal_symmetry + vertical_symmetry) / 2,
        }

    @staticmethod
    def _analyze_region_uniformity(gray: np.ndarray) -> Dict[str, float]:
        """分析区域均匀性"""
        h, w = gray.shape
        region_size = min(h, w) // 8
        region_variances = []

        for i in range(0, h - region_size, region_size):
            for j in range(0, w - region_size, region_size):
                region = gray[i : i + region_size, j : j + region_size]
                region_variances.append(np.var(region))

        return {
            "region_variance_mean": float(np.mean(region_variances)),
            "region_variance_std": float(np.std(region_variances)),
            "uniformity_score": float(1.0 / (1.0 + np.std(region_variances))),
        }

    @staticmethod
    def _detect_jpeg_artifacts(gray: np.ndarray) -> Dict[str, Any]:
        """检测JPEG伪影"""
        h, w = gray.shape
        block_variances = []

        for i in range(0, h - 8, 8):
            for j in range(0, w - 8, 8):
                block = gray[i : i + 8, j : j + 8].astype(np.float32)
                if block.shape == (8, 8):
                    dct_block = cv2.dct(block)
                    high_freq_coeff = dct_block[4:, 4:]
                    block_variances.append(np.var(high_freq_coeff))

        return {
            "block_variance_mean": float(np.mean(block_variances)) if block_variances else 0,
            "block_variance_std": float(np.std(block_variances)) if block_variances else 0,
            "compression_artifacts_score": (
                float(np.std(block_variances) / np.mean(block_variances)) if block_variances and np.mean(block_variances) > 0 else 0
            ),
        }

    @staticmethod
    def _estimate_compression_level(image: np.ndarray) -> Dict[str, Any]:
        """估计压缩程度"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        noise_level = np.std(cv2.GaussianBlur(gray, (3, 3), 0) - gray)

        return {
            "sharpness_score": float(laplacian_var),
            "noise_level": float(noise_level),
            "quality_estimate": "high" if laplacian_var > 1000 else "medium" if laplacian_var > 500 else "low",
        }

    @staticmethod
    def _count_connected_regions(binary_mask: np.ndarray) -> int:
        """计算连通区域数量"""
        binary_uint8 = (binary_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return len([c for c in contours if cv2.contourArea(c) > 50])
