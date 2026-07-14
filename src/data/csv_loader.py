"""CSV upload and validation utilities."""
from pathlib import Path
from typing import BinaryIO

import pandas as pd


class CSVDataLoader:
    """Load and validate CSV files for the BI platform."""

    ALLOWED_EXTENSIONS = {".csv"}

    @staticmethod
    def validate_file(file_name: str) -> bool:
        """Return True if the uploaded file has a supported extension."""
        extension = Path(file_name).suffix.lower()
        return extension in CSVDataLoader.ALLOWED_EXTENSIONS

    @staticmethod
    def load_csv(file_buffer: BinaryIO) -> pd.DataFrame:
        """Load the uploaded CSV into a pandas DataFrame."""
        if not hasattr(file_buffer, "read"):
            raise TypeError("The uploaded file must be a binary file-like object.")

        try:
            dataframe = pd.read_csv(file_buffer)
        except pd.errors.EmptyDataError as error:
            raise ValueError("The CSV file is empty.") from error
        except pd.errors.ParserError as error:
            raise ValueError("The CSV file could not be parsed. Check delimiters and formatting.") from error
        except UnicodeDecodeError as error:
            raise ValueError("Unable to decode the file. Ensure it is UTF-8 encoded.") from error
        except Exception as error:
            raise ValueError("An unexpected error occurred while reading the CSV file.") from error

        if dataframe.empty:
            raise ValueError("The CSV file contains no rows.")

        return dataframe

    @staticmethod
    def preview(dataframe: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
        """Return the first rows of the DataFrame for preview."""
        return dataframe.head(rows)
