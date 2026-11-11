#!/usr/bin/env bash
# Drops the mirrors that do not answer from an apt mirror list
# (apt-transport-mirror(1)).
#
# usage: select-mirrors.sh <mirror-list-file> <codename>
#
# apt moves to the next mirror only after a fetch has failed, so a dead mirror
# costs every index file its full retry budget. One probe per mirror up front
# removes the dead ones. The probe is the `<codename>-security` InRelease
# file, which every mirror the images list serves.
#
# Invariant: the list after the script is the list before it minus the lines
# whose mirror did not answer; when every mirror answered the file is not
# written at all. A kept line is written byte for byte, so the priorities that
# fix the order apt tries the mirrors in, and the other metadata after the
# URI, stay as they were.
#
# A line names a mirror by its first field, fields being separated by spaces,
# tabs or a carriage return. Blank lines and lines whose first field starts
# with `#` name none and are kept. Two URIs that differ only by one trailing
# slash are one mirror and are probed once.
#
# Exit status: 0 when at least one mirror answered; 1 when none answered, with
# the list left unchanged; 2 on a usage error or a missing, empty or
# mirror-less list.

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

# `lines[i]` is the i-th line and `keys[i]` its mirror, empty for a line that
# names none. `|| [[ -n $line ]]` reads a last line that has no newline.
lines=()
keys=()
mirrors=()
declare -A first_written=()
while IFS= read -r line || [[ -n $line ]]; do
  IFS=$' \t\r' read -r first _ <<< "$line"
  key=
  if [[ -n $first && $first != \#* ]]; then
    # A bare "/" would strip to the empty string, which is no array subscript.
    key=${first%/}
    key=${key:-$first}
    if [[ -z ${first_written[$key]+x} ]]; then
      first_written[$key]=$first
      mirrors+=("$key")
    fi
  fi
  lines+=("$line")
  keys+=("$key")
done < "$list"

if [[ ${#mirrors[@]} -eq 0 ]]; then
  echo "mirror list $list names no mirror" >&2
  exit 2
fi

echo "mirror list as found ($list):"
cat "$list"

declare -A answered=()
failed=()
for key in "${mirrors[@]}"; do
  uri=${first_written[$key]}
  started=${EPOCHREALTIME//[.,]/}
  if message=$(curl --silent --show-error --fail --head --location \
    --connect-timeout 5 --max-time 10 --output /dev/null \
    "${key}/dists/${codename}-security/InRelease" 2>&1); then
    reachable=true
  else
    reachable=false
  fi
  elapsed_us=$((10#${EPOCHREALTIME//[.,]/} - 10#$started))
  seconds=$(printf '%d.%03d' $((elapsed_us / 1000000)) $((elapsed_us % 1000000 / 1000)))
  if [[ $reachable == true ]]; then
    echo "reachable $uri ${seconds}s"
    answered[$key]=1
  else
    message=${message//$'\n'/ }
    echo "unreachable $uri ${seconds}s: $message"
    failed+=("$uri")
  fi
done

if [[ ${#answered[@]} -eq 0 ]]; then
  echo "no mirror answered; tried: ${failed[*]}" >&2
  exit 1
fi

if [[ ${#failed[@]} -eq 0 ]]; then
  echo "every mirror answered; the list is left as found ($list)"
  exit 0
fi

kept=()
for i in "${!lines[@]}"; do
  if [[ -z ${keys[i]} || -n ${answered[${keys[i]}]+x} ]]; then
    kept+=("${lines[i]}")
  fi
done

# Written in place so that the file keeps its owner and mode.
printf '%s\n' "${kept[@]}" > "$list"

echo "mirror list as written ($list):"
cat "$list"
