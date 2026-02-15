"""Document parsing utilities using LlamaParse."""
import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import nest_asyncio
from llama_parse import LlamaParse

from core.config import get_config, PROJECT_ROOT

nest_asyncio.apply()
logger = logging.getLogger(__name__)


def parse_pdf_folder(
    folder_path: str, 
    api_key: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Parse all PDFs in a folder using LlamaParse.
    
    Args:
        folder_path: Path to folder containing PDF files.
        api_key: LlamaParse API key (uses config default if not provided).
        
    Returns:
        List of parsed JSON objects from LlamaParse.
    """
    config = get_config()
    
    if api_key is None:
        api_key = config.llama_api_key
        
    if not api_key:
        raise ValueError("LlamaParse API key is required - set LLAMA_API_KEY env var")
    
    parser = LlamaParse(
        result_type="markdown",
        skip_diagonal_text=True,
        fast_mode=False,
        num_workers=9,
        check_interval=10,
        api_key=api_key
    )

    json_objs = []
    folder = Path(folder_path)
    
    if not folder.exists():
        logger.warning(f"PDF folder does not exist: {folder_path}")
        return json_objs

    pdf_files = list(folder.glob("*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}")
    
    for pdf in pdf_files:
        try:
            logger.info(f"Parsing: {pdf.name}")
            json_objs.extend(parser.get_json_result(str(pdf)))
        except Exception as e:
            logger.error(f"Failed to parse {pdf.name}: {e}")
    
    return json_objs


def extract_tables(json_objs: List[Dict[str, Any]]) -> Dict[str, Dict[int, Any]]:
    """
    Extract tables from parsed PDF JSON objects.
    
    Args:
        json_objs: List of parsed JSON objects from LlamaParse.
        
    Returns:
        Dictionary mapping document names to page-table mappings.
    """
    tables = {}

    for obj in json_objs:
        json_list = obj.get('pages', [])
        file_path = obj.get("file_path", "unknown")
        name = Path(file_path).name
        tables[name] = {}

        for json_item in json_list:
            page_num = json_item.get('page', 0)
            for component in json_item.get('items', []):
                if component.get('type') == 'table':
                    tables[name][page_num] = component.get('rows', [])

    return tables


def get_default_pdf_folder() -> Path:
    """Get the default PDF folder path from config."""
    config = get_config()
    return config.data_dir / "unzipped_docs" / "Nutritional Medical Reference"


def get_parsed_tables(folder_path: Optional[str] = None) -> Dict[str, Dict[int, Any]]:
    """
    Get tables from a PDF folder.
    
    Args:
        folder_path: Path to PDF folder. Uses default if not provided.
        
    Returns:
        Dictionary of extracted tables, or empty dict if parsing fails.
    """
    if folder_path is None:
        folder_path = str(get_default_pdf_folder())
    
    folder = Path(folder_path)
    if not folder.exists():
        logger.warning(f"PDF folder not found: {folder_path}")
        return {}
    
    try:
        json_objs = parse_pdf_folder(folder_path)
        return extract_tables(json_objs)
    except Exception as e:
        logger.error(f"Failed to parse PDFs: {e}")
        return {}


# Lazy-loaded tables for backwards compatibility
_tables_cache = None


def _get_tables() -> Dict[str, Dict[int, Any]]:
    """Get cached tables."""
    global _tables_cache
    if _tables_cache is None:
        _tables_cache = get_parsed_tables()
    return _tables_cache


class _LazyTables:
    """Lazy-loading wrapper for tables - avoids parsing at import time."""
    
    def __iter__(self):
        return iter(_get_tables())
    
    def __getitem__(self, key):
        return _get_tables()[key]
    
    def __len__(self):
        return len(_get_tables())
    
    def keys(self):
        return _get_tables().keys()
    
    def values(self):
        return _get_tables().values()
    
    def items(self):
        return _get_tables().items()


# For backwards compatibility - tables are lazy loaded
tables = _LazyTables()
