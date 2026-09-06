#!/usr/bin/env bash
# Deploy the current ember/mantle/entroptics CODE to a peer box and restart its pool fleet.
#
# NOTE: data-defined operators (op.operator.define — sources, compositions) need NO deploy: they
# live in the SHARED store, so both boxes read the same operator artifact and get updates for free.
# This script is only for PRIMITIVE/code changes (a new interpreter kind, a bug fix in a module).
#
#   scripts/deploy-peer.sh builder@<peer-host>              # restart 8 workers
#   WORKERS=12 scripts/deploy-peer.sh builder@<peer-host>   # ... with 12 workers
set -euo pipefail

# The peer is a box on the local network, so there is no default that could be right on
# another checkout. Peer addresses live in `_fleet/peers/<node>/INVENTORY.md`, which is the
# local tier and does not publish. Refuse rather than guess: a deploy aimed at the wrong host
# either fails obscurely or updates a box nobody meant to touch.
PEER="${1:-${PEER:-}}"
if [ -z "$PEER" ]; then
  echo "deploy-peer.sh: no peer given." >&2
  echo "  Pass one as \$1, or set PEER, e.g. PEER=builder@<peer-host> scripts/deploy-peer.sh" >&2
  echo "  Peer addresses are recorded in _fleet/peers/<node>/INVENTORY.md." >&2
  exit 2
fi
KEY="${KEY:-$HOME/.ssh/entroptics_builder_ed25519}"
PORT="${PORT:-2222}"
WORKERS="${WORKERS:-8}"
# This script is `<workspace>/agience-ember/scripts/deploy-peer.sh` and every path below is
# `$REPO/agience/...`, so REPO is the directory CONTAINING the workspace — THREE levels up
# from here: scripts -> agience-ember -> agience -> REPO.
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
SSH="ssh -i $KEY -p $PORT -o ConnectTimeout=15"
EX="--exclude=__pycache__ --exclude=*.pyc"

echo "deploy → $PEER"
tar czf - -C "$REPO/agience/agience-ember/src"        $EX ember      | $SSH "$PEER" "tar xzf - -C ~/genesis/ember-src"
tar czf - -C "$REPO/agience/agience-mantle/src/mantle" $EX db         | $SSH "$PEER" "tar xzf - -C ~/genesis/mantle"
# `agience-beam` was archived 2026-08-05 and ember imports it in 0 modules
# (`tests/test_ember_holds_the_instrument.py` holds that at zero), so no `core` leg is synced.
tar czf - -C "$REPO/agience/entroptics/src"           $EX entroptics | $SSH "$PEER" "tar xzf - -C ~/genesis/entroptics-src"
# The node loops ship here too. Syncing the libraries (ember, mantle/db, entroptics) alone
# leaves node/{ingest,health,content}-loop.py on every box at whatever was last copied by hand — a
# box running an ingest-loop.py with no lease renewer while the fix sits in the repo, and every
# "why is it behaving like the old code" question traces back to this line.
tar czf - -C "$REPO/agience/agience-ember"              $EX node       | $SSH "$PEER" "tar xzf - -C ~/genesis/.node-stage && cp ~/genesis/.node-stage/node/*.py ~/genesis/"
echo "code synced; restarting $WORKERS pool workers on peer"
$SSH "$PEER" "bash ~/genesis/start-workers.sh $WORKERS"
echo "done"
