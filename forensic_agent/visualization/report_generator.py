import os
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import base64

# 导入日志系统
from ..utils.logger import get_logger, log_execution_time, LogContext
from ..utils import ProgressLogger

try:
    import weasyprint

    HAS_WEASYPRINT = True
except ImportError:
    HAS_WEASYPRINT = False

from jinja2 import Template, Environment, FileSystemLoader
import markdown

from .base_visualizer import BaseVisualizer, VisualizationConfig
from ..llm_integration.card_generator import ModelCard, SegmentCard


class ReportFormat(Enum):
    """报告格式枚举"""

    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"


@dataclass
class ReportSection:
    """报告章节数据类"""

    title: str
    content: str
    charts: List[str] = None  # 图表文件路径列表
    order: int = 0

    def __post_init__(self):
        if self.charts is None:
            self.charts = []

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)


@dataclass
class ReportConfig:
    """报告配置"""

    format: ReportFormat = ReportFormat.HTML
    title: str = "模型性能分析报告"
    subtitle: str = ""
    author: str = "AI模型分析系统"
    include_toc: bool = True
    include_charts: bool = True
    include_raw_data: bool = False
    template_dir: Optional[str] = None
    css_style: str = "professional"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "format": self.format.value,
            "title": self.title,
            "subtitle": self.subtitle,
            "author": self.author,
            "include_toc": self.include_toc,
            "include_charts": self.include_charts,
            "include_raw_data": self.include_raw_data,
            "template_dir": self.template_dir,
            "css_style": self.css_style,
        }


class ReportGenerator(BaseVisualizer):
    """报告生成器

    整合各种分析结果生成完整报告
    """

    def __init__(
        self, output_dir: str = "output", vis_config: Optional[VisualizationConfig] = None, report_config: Optional[ReportConfig] = None
    ):
        """初始化报告生成器

        Args:
            output_dir: 输出目录
            vis_config: 可视化配置
            report_config: 报告配置
        """
        super().__init__(output_dir, vis_config)
        self.report_config = report_config or ReportConfig()

        # 重写logger以使用新的日志系统
        self.logger = get_logger(__name__)

        # 创建报告子目录
        self.reports_dir = os.path.join(self.output_dir, "reports")
        os.makedirs(self.reports_dir, exist_ok=True)
        self.logger.info(f"报告输出目录: {self.reports_dir}")

        # 设置模板环境
        self._setup_template_environment()

        # 报告章节列表
        self.sections: List[ReportSection] = []

        self.logger.info(f"报告生成器初始化完成 (格式: {self.report_config.format.value})")

    def _setup_template_environment(self):
        """设置Jinja2模板环境"""
        if self.report_config.template_dir and os.path.exists(self.report_config.template_dir):
            self.template_env = Environment(loader=FileSystemLoader(self.report_config.template_dir))
        else:
            # 使用内置模板
            self.template_env = Environment(loader=FileSystemLoader("."))

    def add_section(self, title: str, content: str, charts: Optional[List[str]] = None, order: Optional[int] = None) -> None:
        """添加报告章节

        Args:
            title: 章节标题
            content: 章节内容
            charts: 图表路径列表
            order: 章节顺序
        """
        if order is None:
            order = len(self.sections)

        section = ReportSection(title=title, content=content, charts=charts or [], order=order)

        self.sections.append(section)
        self.logger.info(f"添加报告章节: {title}")

    def add_model_card_section(self, model_card: ModelCard, card_image_path: Optional[str] = None) -> None:
        """添加模型卡片章节

        Args:
            model_card: 模型卡片
            card_image_path: 卡片图片路径
        """
        section_title = f"模型分析: {model_card.model_name}"
        section_content = model_card.card_content

        charts = []
        if card_image_path and os.path.exists(card_image_path):
            charts.append(card_image_path)

        self.add_section(section_title, section_content, charts)

    def add_segment_cards_section(self, segment_cards: List[SegmentCard], card_images: Optional[List[str]] = None) -> None:
        """添加内容类型片段卡片章节

        Args:
            segment_cards: 片段卡片列表
            card_images: 卡片图片路径列表
        """
        if not segment_cards:
            return

        section_title = "内容类型分析"

        # 合并所有片段卡片内容
        content_parts = []
        for i, card in enumerate(segment_cards):
            content_parts.append(f"## {card.segment_name}")
            content_parts.append(card.card_content)
            content_parts.append("")  # 空行分隔

        section_content = "\n".join(content_parts)

        # 收集图表路径
        charts = card_images or []

        self.add_section(section_title, section_content, charts)

    def add_statistical_results_section(
        self, statistical_results: Dict[str, Any], interpretation: Optional[str] = None, charts: Optional[List[str]] = None
    ) -> None:
        """添加统计结果章节

        Args:
            statistical_results: 统计结果
            interpretation: 统计解释
            charts: 相关图表路径
        """
        section_title = "统计分析结果"

        # 格式化统计结果
        content_parts = ["## 统计检验摘要\n"]

        for test_name, result in statistical_results.items():
            content_parts.append(f"### {test_name}")

            if isinstance(result, dict):
                for key, value in result.items():
                    if key == "p_value":
                        content_parts.append(f"- **p值**: {value:.6f}")
                    elif key == "statistic":
                        content_parts.append(f"- **统计量**: {value:.4f}")
                    elif key == "effect_size":
                        content_parts.append(f"- **效应量**: {value:.4f}")
                    elif key == "confidence_interval":
                        if isinstance(value, (list, tuple)) and len(value) == 2:
                            content_parts.append(f"- **置信区间**: [{value[0]:.4f}, {value[1]:.4f}]")
            content_parts.append("")

        # 添加解释（如果有）
        if interpretation:
            content_parts.extend(["## 统计结果解释", "", interpretation])

        section_content = "\n".join(content_parts)

        self.add_section(section_title, section_content, charts or [])

    def add_visualization_section(self, title: str, description: str, chart_paths: List[str]) -> None:
        """添加可视化章节

        Args:
            title: 章节标题
            description: 章节描述
            chart_paths: 图表路径列表
        """
        content = f"{description}\n\n"

        # 为每个图表添加引用
        for i, chart_path in enumerate(chart_paths, 1):
            chart_name = os.path.splitext(os.path.basename(chart_path))[0]
            content += f"图表 {i}: {chart_name}\n\n"

        self.add_section(title, content, chart_paths)

    @log_execution_time
    def generate_html_report(self, filename: str = "analysis_report.html") -> str:
        """生成HTML报告

        Args:
            filename: 输出文件名

        Returns:
            str: 生成的文件路径
        """
        with LogContext(f"HTML报告生成: {filename}", level="INFO"):
            self.logger.info(f"开始生成HTML报告: {filename}")
            self.logger.info(f"报告包含 {len(self.sections)} 个章节")

            # 排序章节
            sorted_sections = sorted(self.sections, key=lambda x: x.order)

            # 统计图表数量
            total_charts = sum(len(section.charts) for section in sorted_sections)
            self.logger.info(f"报告包含 {total_charts} 个图表")

            # 准备模板数据
            with LogContext("模板数据准备", level="DEBUG"):
                template_data = {
                    "title": self.report_config.title,
                    "subtitle": self.report_config.subtitle,
                    "author": self.report_config.author,
                    "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "sections": sorted_sections,
                    "include_toc": self.report_config.include_toc,
                    "include_charts": self.report_config.include_charts,
                    "css_style": self.report_config.css_style,
                }

                self.logger.debug(f"模板配置: 包含目录={self.report_config.include_toc}, 包含图表={self.report_config.include_charts}")

            # 渲染HTML模板
            with LogContext("HTML模板渲染", level="DEBUG"):
                html_content = self._render_html_template(template_data)
                self.logger.debug(f"HTML内容长度: {len(html_content)} 字符")

            # 保存文件
            with LogContext("文件保存", level="DEBUG"):
                filepath = os.path.join(self.reports_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(html_content)

                file_size = os.path.getsize(filepath) / 1024  # KB
                self.logger.info(f"HTML报告已保存: {filepath} ({file_size:.1f} KB)")

            return filepath

    def generate_pdf_report(self, filename: str = "analysis_report.pdf") -> str:
        """生成PDF报告

        Args:
            filename: 输出文件名

        Returns:
            str: 生成的文件路径
        """
        if not HAS_WEASYPRINT:
            self.logger.warning("WeasyPrint未安装，无法生成PDF报告")
            return ""

        self.logger.info("生成PDF报告")

        # 先生成HTML内容
        html_content = self._render_html_template(
            {
                "title": self.report_config.title,
                "subtitle": self.report_config.subtitle,
                "author": self.report_config.author,
                "generation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sections": sorted(self.sections, key=lambda x: x.order),
                "include_toc": self.report_config.include_toc,
                "include_charts": self.report_config.include_charts,
                "css_style": "pdf",  # PDF专用样式
            }
        )

        # 转换为PDF
        try:
            filepath = os.path.join(self.reports_dir, filename)
            weasyprint.HTML(string=html_content, base_url=self.reports_dir).write_pdf(filepath)
            self.logger.info(f"PDF报告已保存: {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"PDF生成失败: {e}")
            return ""

    def generate_markdown_report(self, filename: str = "analysis_report.md") -> str:
        """生成Markdown报告

        Args:
            filename: 输出文件名

        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成Markdown报告")

        # 排序章节
        sorted_sections = sorted(self.sections, key=lambda x: x.order)

        # 构建Markdown内容
        md_parts = []

        # 标题部分
        md_parts.extend(
            [
                f"# {self.report_config.title}",
                "",
                f"**作者**: {self.report_config.author}",
                f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                "",
            ]
        )

        if self.report_config.subtitle:
            md_parts.extend([f"## {self.report_config.subtitle}", ""])

        # 目录（如果需要）
        if self.report_config.include_toc:
            md_parts.extend(["## 目录", ""])

            for section in sorted_sections:
                md_parts.append(f"- [{section.title}](#{section.title.lower().replace(' ', '-')})")
            md_parts.append("")

        # 章节内容
        for section in sorted_sections:
            md_parts.extend([f"# {section.title}", "", section.content, ""])

            # 添加图表引用（如果需要）
            if self.report_config.include_charts and section.charts:
                md_parts.extend(["## 相关图表", ""])

                for chart_path in section.charts:
                    chart_name = os.path.basename(chart_path)
                    # 使用相对路径
                    rel_path = os.path.relpath(chart_path, self.reports_dir)
                    md_parts.append(f"![{chart_name}]({rel_path})")

                md_parts.append("")

        # 保存文件
        markdown_content = "\n".join(md_parts)
        filepath = os.path.join(self.reports_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(markdown_content)

        self.logger.info(f"Markdown报告已保存: {filepath}")
        return filepath

    def generate_json_report(self, filename: str = "analysis_report.json") -> str:
        """生成JSON格式报告

        Args:
            filename: 输出文件名

        Returns:
            str: 生成的文件路径
        """
        self.logger.info("生成JSON报告")

        # 构建JSON数据
        report_data = {
            "metadata": {
                "title": self.report_config.title,
                "subtitle": self.report_config.subtitle,
                "author": self.report_config.author,
                "generation_time": datetime.now().isoformat(),
                "format_version": "1.0",
            },
            "config": self.report_config.to_dict(),
            "sections": [section.to_dict() for section in sorted(self.sections, key=lambda x: x.order)],
        }

        # 保存文件
        filepath = os.path.join(self.reports_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"JSON报告已保存: {filepath}")
        return filepath

    @log_execution_time
    def generate_comprehensive_report(self, base_filename: str = "analysis_report") -> Dict[str, str]:
        """生成所有格式的综合报告

        Args:
            base_filename: 基础文件名

        Returns:
            Dict[str, str]: 格式到文件路径的映射
        """
        with LogContext(f"综合报告生成: {base_filename}", level="INFO"):
            self.logger.info(f"开始生成所有格式的综合报告: {base_filename}")

            report_formats = ["HTML", "PDF", "Markdown", "JSON"]
            progress_logger = ProgressLogger(len(report_formats), "报告格式生成", __name__)

            results = {}

            # HTML报告
            with LogContext("HTML报告生成", level="DEBUG"):
                progress_logger.update(1)
                html_path = self.generate_html_report(f"{base_filename}.html")
                if html_path:
                    results["html"] = html_path
                    self.logger.info("HTML报告生成成功")
                else:
                    self.logger.warning("HTML报告生成失败")

            # PDF报告
            with LogContext("PDF报告生成", level="DEBUG"):
                progress_logger.update(1)
                pdf_path = self.generate_pdf_report(f"{base_filename}.pdf")
                if pdf_path:
                    results["pdf"] = pdf_path
                    self.logger.info("PDF报告生成成功")
                else:
                    self.logger.warning("PDF报告生成失败（可能缺少WeasyPrint）")

            # Markdown报告
            with LogContext("Markdown报告生成", level="DEBUG"):
                progress_logger.update(1)
                md_path = self.generate_markdown_report(f"{base_filename}.md")
                if md_path:
                    results["markdown"] = md_path
                    self.logger.info("Markdown报告生成成功")
                else:
                    self.logger.warning("Markdown报告生成失败")

            # JSON报告
            with LogContext("JSON报告生成", level="DEBUG"):
                progress_logger.update(1)
                json_path = self.generate_json_report(f"{base_filename}.json")
                if json_path:
                    results["json"] = json_path
                    self.logger.info("JSON报告生成成功")
                else:
                    self.logger.warning("JSON报告生成失败")

            self.logger.info(f"综合报告生成完成: {len(results)}/{len(report_formats)} 格式成功")

            # 记录生成的报告文件
            for format_name, file_path in results.items():
                file_size = os.path.getsize(file_path) / 1024  # KB
                self.logger.info(f"{format_name.upper()}报告: {file_path} ({file_size:.1f} KB)")

            return results

    def _render_html_template(self, data: Dict[str, Any]) -> str:
        """渲染HTML模板

        Args:
            data: 模板数据

        Returns:
            str: 渲染后的HTML内容
        """
        # 内置HTML模板
        template_content = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        {{ css_styles }}
    </style>
</head>
<body>
    <div class="container">
        <header class="report-header">
            <h1>{{ title }}</h1>
            {% if subtitle %}
            <h2>{{ subtitle }}</h2>
            {% endif %}
            <div class="report-meta">
                <p><strong>作者:</strong> {{ author }}</p>
                <p><strong>生成时间:</strong> {{ generation_time }}</p>
            </div>
        </header>
        
        {% if include_toc and sections|length > 1 %}
        <nav class="toc">
            <h2>目录</h2>
            <ul>
            {% for section in sections %}
                <li><a href="#section-{{ loop.index }}">{{ section.title }}</a></li>
            {% endfor %}
            </ul>
        </nav>
        {% endif %}
        
        <main>
            {% for section in sections %}
            <section id="section-{{ loop.index }}" class="report-section">
                <h2>{{ section.title }}</h2>
                <div class="section-content">
                    {{ section.content | markdown }}
                </div>
                
                {% if include_charts and section.charts %}
                <div class="section-charts">
                    <h3>相关图表</h3>
                    {% for chart_path in section.charts %}
                    <div class="chart-container">
                        <img src="{{ chart_path | chart_to_base64 }}" alt="Chart" class="chart-image">
                        <p class="chart-caption">{{ chart_path | basename }}</p>
                    </div>
                    {% endfor %}
                </div>
                {% endif %}
            </section>
            {% endfor %}
        </main>
        
        <footer class="report-footer">
            <p>此报告由AI模型分析系统自动生成</p>
        </footer>
    </div>
</body>
</html>
        """

        # CSS样式
        css_styles = self._get_css_styles(data.get("css_style", "professional"))

        # 创建模板并渲染
        template = Template(template_content)
        template.globals["markdown"] = self._markdown_filter
        template.globals["chart_to_base64"] = self._chart_to_base64_filter
        template.globals["basename"] = os.path.basename

        data_with_css = dict(data)
        data_with_css["css_styles"] = css_styles

        return template.render(**data_with_css)

    def _get_css_styles(self, style_name: str) -> str:
        """获取CSS样式

        Args:
            style_name: 样式名称

        Returns:
            str: CSS样式内容
        """
        styles = {
            "professional": """
                body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; margin: 0; padding: 0; color: #333; }
                .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
                .report-header { text-align: center; border-bottom: 2px solid #2E86AB; padding-bottom: 20px; margin-bottom: 30px; }
                .report-header h1 { color: #2E86AB; margin-bottom: 10px; }
                .report-header h2 { color: #666; font-weight: normal; }
                .report-meta { color: #666; font-size: 14px; }
                .toc { background: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }
                .toc ul { list-style-type: none; padding: 0; }
                .toc li { padding: 5px 0; }
                .toc a { text-decoration: none; color: #2E86AB; }
                .report-section { margin-bottom: 40px; }
                .report-section h2 { color: #2E86AB; border-bottom: 1px solid #ddd; padding-bottom: 10px; }
                .section-content { margin: 20px 0; }
                .section-charts { margin-top: 30px; }
                .chart-container { text-align: center; margin: 20px 0; }
                .chart-image { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
                .chart-caption { font-style: italic; color: #666; margin-top: 10px; }
                .report-footer { text-align: center; color: #666; border-top: 1px solid #ddd; padding-top: 20px; margin-top: 40px; }
            """,
            "pdf": """
                body { font-family: 'DejaVu Sans', sans-serif; line-height: 1.6; margin: 20px; color: #333; }
                .container { max-width: none; margin: 0; padding: 0; }
                .report-header { text-align: center; border-bottom: 2px solid #2E86AB; padding-bottom: 20px; margin-bottom: 30px; }
                .report-header h1 { color: #2E86AB; margin-bottom: 10px; font-size: 24px; }
                .report-header h2 { color: #666; font-weight: normal; font-size: 18px; }
                .report-meta { color: #666; font-size: 12px; }
                .toc { background: #f8f9fa; padding: 15px; margin-bottom: 25px; page-break-inside: avoid; }
                .toc ul { list-style-type: none; padding: 0; }
                .toc li { padding: 3px 0; }
                .report-section { margin-bottom: 30px; page-break-inside: avoid; }
                .report-section h2 { color: #2E86AB; border-bottom: 1px solid #ddd; padding-bottom: 8px; font-size: 18px; }
                .section-content { margin: 15px 0; }
                .section-charts { margin-top: 20px; }
                .chart-container { text-align: center; margin: 15px 0; page-break-inside: avoid; }
                .chart-image { max-width: 100%; height: auto; border: 1px solid #ddd; }
                .chart-caption { font-style: italic; color: #666; margin-top: 8px; font-size: 12px; }
                .report-footer { text-align: center; color: #666; border-top: 1px solid #ddd; padding-top: 15px; margin-top: 30px; }
            """,
        }

        return styles.get(style_name, styles["professional"])

    def _markdown_filter(self, text: str) -> str:
        """Markdown过滤器"""
        return markdown.markdown(text, extensions=["tables", "fenced_code"])

    def _chart_to_base64_filter(self, chart_path: str) -> str:
        """将图表转换为base64数据URI"""
        try:
            if os.path.exists(chart_path):
                with open(chart_path, "rb") as f:
                    image_data = f.read()
                    b64_data = base64.b64encode(image_data).decode("utf-8")

                    # 检测图片格式
                    ext = os.path.splitext(chart_path)[1].lower()
                    if ext == ".png":
                        mime_type = "image/png"
                    elif ext in [".jpg", ".jpeg"]:
                        mime_type = "image/jpeg"
                    elif ext == ".svg":
                        mime_type = "image/svg+xml"
                    else:
                        mime_type = "image/png"

                    return f"data:{mime_type};base64,{b64_data}"
        except Exception as e:
            self.logger.warning(f"无法转换图表为base64: {chart_path}, 错误: {e}")

        return chart_path  # 返回原路径作为fallback

    def clear_sections(self):
        """清空所有章节"""
        self.sections.clear()
        self.logger.info("已清空所有报告章节")

    def generate(self, data: Dict[str, Any], **kwargs) -> str:
        """生成报告（实现抽象方法）

        Args:
            data: 输入数据
            **kwargs: 其他参数

        Returns:
            str: 生成的文件路径
        """
        format_type = kwargs.get("format", self.report_config.format)
        filename = kwargs.get("filename", "analysis_report")

        if format_type == ReportFormat.HTML or format_type.value == "html":
            return self.generate_html_report(f"{filename}.html")
        elif format_type == ReportFormat.PDF or format_type.value == "pdf":
            return self.generate_pdf_report(f"{filename}.pdf")
        elif format_type == ReportFormat.MARKDOWN or format_type.value == "markdown":
            return self.generate_markdown_report(f"{filename}.md")
        elif format_type == ReportFormat.JSON or format_type.value == "json":
            return self.generate_json_report(f"{filename}.json")
        else:
            # 默认生成HTML
            return self.generate_html_report(f"{filename}.html")


# 便捷函数
def create_report_generator(
    output_dir: str = "output", vis_config: Optional[VisualizationConfig] = None, report_config: Optional[ReportConfig] = None, **kwargs
) -> ReportGenerator:
    """创建报告生成器的便捷函数"""
    return ReportGenerator(output_dir=output_dir, vis_config=vis_config, report_config=report_config)
