"""Result persistence outside the timed numerical fit."""
from pathlib import Path
import json, math, time
import numpy as np

ARRAY_FIELDS=frozenset(('W','W_raw','W_column','graph','_state'))

def metadata(value):
    """Keep nonfinite diagnostic values as null, never manufacture zero weights."""
    if isinstance(value,np.generic):return metadata(value.item())
    if isinstance(value,float):return value if math.isfinite(value) else None
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,dict):return {str(k):metadata(v) for k,v in value.items() if k not in ARRAY_FIELDS}
    if isinstance(value,(tuple,list)):return [metadata(v) for v in value]
    return value


def save_result(result,directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    arrays={k:result[k] for k in ('W','W_column','graph') if result.get(k) is not None}
    if arrays:np.savez_compressed(directory/'weights.npz',**arrays)
    document=metadata(result)
    document['output_io_seconds']=time.perf_counter()-start
    document['output_note']='weights are separate; nonfinite scalar diagnostics are JSON null; output serialization excluded from hot-fit timer'
    (directory/'result.json').write_text(json.dumps(document,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return directory/'result.json'
