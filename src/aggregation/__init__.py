"""聚合器模块入口。"""
from src.aggregation.aggregator import aggregate, detect_doc_type
from src.aggregation.schemas import MergedCase, MergedDocument

__all__ = ["aggregate", "detect_doc_type", "MergedCase", "MergedDocument"]
