from pathlib import Path
import os

data_dir = Path(os.getenv("DATADIR"))
test_pdf_path = data_dir / "test" / "test_01.pdf"
