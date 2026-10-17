import json
import sys
from pathlib import Path
import numpy as np
import polars as pl
from .strategy import evaluate_in_process

if __name__=='__main__':
    directory=Path(sys.argv[1])
    try:
        features=len(sys.argv)>3 and sys.argv[3]=='features'
        values=evaluate_in_process((directory/'strategy.py').read_text(encoding='utf-8'),pl.read_parquet(directory/'input.parquet'),sys.argv[2]=='1',features)
        if features:
            values.write_parquet(directory/'features.parquet')
        else:
            np.save(directory/'signal.npy',values,allow_pickle=False)
    except Exception as error:
        (directory/'error.json').write_text(json.dumps({'error':str(error)}),encoding='utf-8')
        raise SystemExit(1)
