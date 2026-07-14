#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Report roots written by the run scripts (mode is now a single|multi script arg):
#   e2e      -> e2e/report1                (single + multi share this root)
#   overhead -> overhead/report            (single)
#            -> overhead/report_multigpu   (multi)
rm -rf "${SCRIPT_DIR}/e2e/report1"
rm -rf "${SCRIPT_DIR}/overhead/report"
rm -rf "${SCRIPT_DIR}/overhead/report_multigpu"

echo "Cleaned vLLM e2e + overhead reports."
