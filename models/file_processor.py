import base64
import docx
import io
import json
import mimetypes
import pandas as pd
import os

class FileProcessor:
    """
    Process files
    """

    SUPPORTED_TEXT_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".json",
        ".html", ".css", ".xml", ".yaml", ".yml",
        ".md", ".txt", ".sql", ".sh", ".env"
    }

    IMAGE_MIME_TYPES = {
        "image/jpeg", "image/png", "image/gif", "image/webp"
    }

    @classmethod
    def process(cls, file_path: str) -> dict:
        """
        Main function - send the file to correct function according to its type
        :param file_path: path to file
        :return dict of content
        """
        filename = os.path.basename(file_path)
        ext = os.path.splitext(filename)[1].lower()

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        if ext in cls.SUPPORTED_TEXT_EXTENSIONS:
            return cls._process_text(file_bytes, filename)

        elif ext == ".ipynb":
            return cls._process_ipynb(file_bytes, filename)

        elif ext == ".pdf":
            return cls._process_pdf(file_bytes)

        elif ext == ".docx":
            return cls._process_docx(file_bytes, filename)

        elif ext in (".xlsx", ".xls"):
            return cls._process_xlsx(file_bytes, filename)
        else:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type in cls.IMAGE_MIME_TYPES:
                return cls._process_image(file_bytes, mime_type)
            else:
                raise ValueError(f"Unsupported file type for '{filename}'. ")

    @classmethod
    def _process_text(cls, file_bytes: bytes, filename: str) -> dict:
        """
        Process text files
        :param file_bytes: input stream
        :param filename: file name
        :return: dict with file data
        """
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1")

        ext = os.path.splitext(filename)[1].lstrip(".")
        return {
            "type": "text_content",
            "filename": filename,
            "text": f"```{ext}\n{content}\n```"
        }

    @classmethod
    def _process_ipynb(cls, file_bytes: bytes, filename: str) -> dict:
        """
        Process notebook files
        :param file_bytes: input stream
        :param filename: file name
        :return: dict with file data
        """
        notebook = json.loads(file_bytes.decode("utf-8"))
        parts = [f"# Jupyter Notebook: {filename}\n"]

        for i, cell in enumerate(notebook.get("cells", []), 1):
            cell_type = cell.get("cell_type", "")
            source = "".join(cell.get("source", []))

            if cell_type == "code":
                parts.append(f"## Cell {i} [code]\n```python\n{source}\n```")
                # Add outputs to get complete data
                outputs = cell.get("outputs", [])
                if outputs:
                    output_texts = []
                    for out in outputs:
                        if "text" in out:
                            output_texts.append("".join(out["text"]))
                        elif "traceback" in out:
                            output_texts.append("\n".join(out["traceback"]))
                    if output_texts:
                        parts.append(f"**Output:**\n```\n{''.join(output_texts)}\n```")

            elif cell_type == "markdown":
                parts.append(f"## Cell {i} [markdown]\n{source}")

        return {
            "type": "text_content",
            "filename": filename,
            "text": "\n\n".join(parts)
        }

    @classmethod
    def _process_pdf(cls, file_bytes: bytes) -> dict:
        """
        Process PDF files
        :param file_bytes: input stream
        :param filename: file name
        :return: dict with file data
        """
        return {
            "type": "pdf",
            "data": base64.standard_b64encode(file_bytes).decode("utf-8")
        }

    @classmethod
    def _process_docx(cls, file_bytes: bytes, filename: str) -> dict:
        """
        Process Word .docx files
        :param file_bytes: input stream
        :param filename: file name
        :return: dict with file data
        """
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        # process also document tables
        table_texts = []
        for table in doc.tables:
            rows = []
            for row in table.rows:
                rows.append(" | ".join(cell.text.strip() for cell in row.cells))
            table_texts.append("\n".join(rows))

        content = "\n\n".join(paragraphs)
        if table_texts:
            content += "\n\n## Tables\n\n" + "\n\n---\n\n".join(table_texts)

        return {
            "type": "text_content",
            "filename": filename,
            "text": content
        }

    @classmethod
    def _process_xlsx(cls, file_bytes: bytes, filename: str) -> dict:
        """
        Process Excel .xlsx files
        :param file_bytes: input stream
        :param filename: file name
        :return: dict with file data
        """
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        parts = [f"# Excel file: {filename}"]

        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)
            # generate as Markdown tables - easily readable for LLM
            parts.append(
                f"## Sheet: {sheet_name}\n\n{df.to_markdown(index=False)}"
            )

        return {
            "type": "text_content",
            "filename": filename,
            "text": "\n\n".join(parts)
        }

    @classmethod
    def _process_image(cls, file_bytes: bytes, mime_type: str) -> dict:
        """
        Process image files
        :param file_bytes: input stream
        :param filename: file name
        :return: dict with file data
        """
        return {
            "type": "image",
            "media_type": mime_type,
            "data": base64.standard_b64encode(file_bytes).decode("utf-8")
        }