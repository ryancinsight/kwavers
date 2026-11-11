#!/usr/bin/env bash
# Rewrites an apt mirror list (apt-transport-mirror(1)) to the mirrors that
# answer, in the order listed.
#
# usage: select-mirrors.sh <mirror-list-file> <codename>
#
# apt moves to the next mirror only after a fetch has failed, so a dead mirror
# costs every index file its full retry budget. One probe per mirror up front
# removes the dead ones. The probe is the `<codename>-security` InRelease
# file, which every mirror the images list serves.
#
# Exit status: 0 when at least one mirror answered and the list was rewritten;
# 1 when none answered, with the list left unchanged; 2 on a usage error or a
# missing or empty list.

set -euo pipefail

if [[ $# -ne 2 || -z $2 ]]; then
  echo "usage: select-mirrors.sh <mirror-list-file> <codename>" >&2
  exit 2
fi
list=$1
codename=$2

if [[ ! -s $list ]]; then
  echo "mirror list $list is missing or empty" >&2
  exit 2
fi

# The first field of a line is the URI, the rest is metadata; a repeated URI
# is probed once and keeps its first position.
mirrors=()
declare -A seen=()
while read -r uri _; do
  [[ -n $uri && $uri != \#* && -z ${seen[$uri]+x} ]] || continue
  seen[$uri]=1
  mirrors+=("$uri")
done < "$list"

if [[ ${#mirrors[@]} -eq 0 ]]; then
  echo "mirror list $list names no mirror" >&2
  exit 2
fi

echo "mirror list as found ($list):"
cat "$list"

kept=()
failed=()
for uri in "${mirrors[@]}"; do
  started=${EPOCHREALTIME//[.,]/}
  if message=$(curl --silent --show-error --fail --head --location \
    --connect-timeout 5 --max-time 10 --output /dev/null \
    "${uri%/}/dists/${codename}-security/InRelease" 2>&1); then
    reachable=true
  else
    reachable=false
  fi
  elapsed_us=$((10#${EPOCHREALTIME//[.,]/} - 10#$started))
  seconds=$(printf '%d.%03d' $((elapsed_us / 1000000)) $((elapsed_us % 1000000 / 1000)))
  if [[ $reachable == true ]]; then
    echo "reachable $uri ${seconds}s"
    kept+=("$uri")
  else
    message=${message//$'\n'/ }
    echo "unreachable $uri ${seconds}s: $message"
    failed+=("$uri")
  fi
done

if [[ ${#kept[@]} -eq 0 ]]; then
  echo "no mirror answered; tried: ${failed[*]}" >&2
  exit 1
fi

# Written in place so that the file keeps its owner and mode.
rewritten=""
priority=0
for uri in "${kept[@]}"; do
  priority=$((priority + 1))
  rewritten+=$(printf '%s\tpriority:%d' "$uri" "$priority")$'\n'
done
printf '%s' "$rewritten" > "$list"

echo "mirror list as written ($list):"
cat "$list"
