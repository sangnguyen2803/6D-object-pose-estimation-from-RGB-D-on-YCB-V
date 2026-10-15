"""Audit the current YCB-V dataset."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from clutterpick.common import load_config,resolve
from clutterpick.audit import audit_dataset
if __name__=='__main__':
    print(audit_dataset(load_config(),resolve('reports/ycbv/data_audit.json')))

