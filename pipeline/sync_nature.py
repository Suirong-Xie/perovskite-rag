#!/usr/bin/env python3
import os, glob, shutil, re
from datetime import datetime

JOURNALS_PDF = '/data/data/pkb/01_raw_data/journals_pdf'
PAPERS_PDF = '/data/data/pkb/01_raw_data/papers_pdf'
extract_year = re.compile(r'_(20\d{2})_')

for src in glob.glob(os.path.join(JOURNALS_PDF, '**/*.pdf'), recursive=True):
    fname = os.path.basename(src)
    m = extract_year.search(fname)
    year = m.group(1) if m else str(datetime.now().year)
    month = datetime.now().strftime('%m')
    dest = os.path.join(PAPERS_PDF, year, month, fname)
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(src, dest)

all_pdfs = glob.glob(os.path.join(PAPERS_PDF, '**/*.pdf'), recursive=True)
print(f'Synced. Total PDFs in papers_pdf: {len(all_pdfs)}')
