#!/usr/bin/env bash
# The complete e2e of Ember — one green button. The lattice opens in-process, so there is
# no substrate to provision first.
#   1. unit tests (distill engine, arithmetic, reason)
#   2. lexical e2e     (WordNet keyed dictionary)
#   3. copilot e2e     (OpenAI-compatible /v1 over HTTP -> grounded, streaming, ungrounded)
#
# Usage:  bash agience-ember/tests/e2e/run-all.sh
set -euo pipefail
cd "$(dirname "$0")/../../.."       # -> the workspace root (Repos/agience)
export PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_PROGRESS_BARS=1
# A box that wants the (large) model cache off the system drive names a volume in
# EMBER_CACHE_DIR. The default is home-relative so a fresh checkout runs without one.
export EMBER_CACHE_DIR="${EMBER_CACHE_DIR:-$HOME/.cache/agience/ember}"
# mantle's outer src: it is imported as a package (`mantle.db.…`), so its modules import each
# other package-qualified and the inner `src/mantle` path only half-loads.
export PYTHONPATH="agience-ember/src;agience-mantle/src;entroptics/src"

echo "== 1. unit tests (distill, arithmetic+category, reason) =="
python -m pytest agience-ember/src/ember/test_engine_distill.py \
                 agience-ember/src/ember/test_arithmetic.py \
                 agience-ember/src/ember/test_reason.py -q

echo "== 2. lexical e2e (WordNet keyed dictionary) =="
python agience-ember/e2e/test_lexical_e2e.py 2>&1 | grep -vE "Fetching|it/s"

echo "== 3. copilot e2e (multi-domain: arithmetic + lexical + refusal over /v1) =="
python agience-ember/tests/e2e/test_copilot_e2e.py 2>&1 | grep -vE "Fetching|it/s"

echo ""
echo "===================== EMBER E2E: ALL GREEN ====================="
