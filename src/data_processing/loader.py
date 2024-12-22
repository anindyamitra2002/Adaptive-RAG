from pathlib import Path
from typing import List, Union
import logging
from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument
from langchain_core.document_loaders import BaseLoader
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat, ConversionStatus
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    EasyOcrOptions
)

logging.basicConfig(level=logging.INFO)
_log = logging.getLogger(__name__)

@dataclass
class ProcessingResult:
    """Store results of document processing"""
    success_count: int = 0
    failure_count: int = 0
    partial_success_count: int = 0
    failed_files: List[str] = None

    def __post_init__(self):
        if self.failed_files is None:
            self.failed_files = []

class MultiFormatDocumentLoader(BaseLoader):
    """Loader for multiple document formats that converts to LangChain documents"""
    
    def __init__(
        self,
        file_paths: Union[str, List[str]],
        enable_ocr: bool = True,
        enable_tables: bool = True
    ):
        self._file_paths = [file_paths] if isinstance(file_paths, str) else file_paths
        self._enable_ocr = enable_ocr
        self._enable_tables = enable_tables
        self._converter = self._setup_converter()
        
    def _setup_converter(self):
        """Set up the document converter with appropriate options"""
        # Configure pipeline options
        pipeline_options = PdfPipelineOptions(do_ocr=False, do_table_structure=False, ocr_options=EasyOcrOptions(
                force_full_page_ocr=True
            ))
        if self._enable_ocr:
            pipeline_options.do_ocr = True
        if self._enable_tables:
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = True

        # Create converter with supported formats
        return DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
                InputFormat.IMAGE,
                InputFormat.DOCX,
                InputFormat.HTML,
                InputFormat.PPTX,
                InputFormat.ASCIIDOC,
                InputFormat.MD,
            ],
            format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )}
        )

    def lazy_load(self):
        """Convert documents and yield LangChain documents"""
        results = ProcessingResult()
        
        for file_path in self._file_paths:
            try:
                path = Path(file_path)
                if not path.exists():
                    _log.warning(f"File not found: {file_path}")
                    results.failure_count += 1
                    results.failed_files.append(file_path)
                    continue

                conversion_result = self._converter.convert(path)
                
                if conversion_result.status == ConversionStatus.SUCCESS:
                    results.success_count += 1
                    text = conversion_result.document.export_to_markdown()
                    metadata = {
                        'source': str(path),
                        'file_type': path.suffix,
                    }
                    yield LCDocument(
                        page_content=text,
                        metadata=metadata
                    )
                elif conversion_result.status == ConversionStatus.PARTIAL_SUCCESS:
                    results.partial_success_count += 1
                    _log.warning(f"Partial conversion for {file_path}")
                    text = conversion_result.document.export_to_markdown()
                    metadata = {
                        'source': str(path),
                        'file_type': path.suffix,
                        'conversion_status': 'partial'
                    }
                    yield LCDocument(
                        page_content=text,
                        metadata=metadata
                    )
                else:
                    results.failure_count += 1
                    results.failed_files.append(file_path)
                    _log.error(f"Failed to convert {file_path}")
                    
            except Exception as e:
                _log.error(f"Error processing {file_path}: {str(e)}")
                results.failure_count += 1
                results.failed_files.append(file_path)

        # Log final results
        total = results.success_count + results.partial_success_count + results.failure_count
        _log.info(
            f"Processed {total} documents:\n"
            f"- Successfully converted: {results.success_count}\n"
            f"- Partially converted: {results.partial_success_count}\n"
            f"- Failed: {results.failure_count}"
        )
        if results.failed_files:
            _log.info("Failed files:")
            for file in results.failed_files:
                _log.info(f"- {file}")
                
                
if __name__ == '__main__':
    # Load documents from a list of file paths
    loader = MultiFormatDocumentLoader(
        file_paths=[
            # './data/2404.19756v1.pdf',
            # './data/OD429347375590223100.pdf',
            './data/Project Report Format.docx',
            # './data/UNIT 2 GENDER BASED VIOLENCE.pptx'
        ],
        enable_ocr=False,
        enable_tables=True
    )
    for doc in loader.lazy_load():
        print(doc.page_content)
        print(doc.metadata)
        # save document in .md file 
        with open('output.md', 'w') as f:
            f.write(doc.page_content)