"""Audit the current YCB-V dataset."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from clutterpick.data.audit import audit_dataset
from clutterpick.utils.common import load_config, resolve

if __name__ == "__main__":
    print(audit_dataset(load_config(), resolve("reports/ycbv/data_audit.json")))
