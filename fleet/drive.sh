#!/bin/bash
# Put the code and the credentials on one instance, then start it.
#
# The uploader starts BEFORE the run. In the first five-tile run it was
# installed after the pipeline had already begun, so for several minutes the
# results existed only on a disk that terminate-on-shutdown would destroy.
#
# Every remote sequence goes through a script rather than an expanded shell
# variable. A bare `$SSH` under zsh does not word-split, which once broke a run
# silently for 600 seconds while the poll loop sent stderr to /dev/null.
set -euo pipefail
MANIFEST="${1:?run manifest json}"
TILE="${2:?tile id}"

read -r NAME PEM IP COMMIT BUCKET ART RUNS UPROF < <(
  python3 - "$MANIFEST" "$TILE" <<'PY'
import json, sys
run = json.load(open(sys.argv[1])); tile = sys.argv[2]
e = next(i for i in run["instances"] if i["tile"] == tile)
c = run["config"]; s = c["storage"]
print(e["name"], e["pem"], e["ip"], run["commit"], s["bucket"],
      f's3://{s["bucket"]}/{s["artifacts_prefix"]}', s["runs_prefix"],
      s["upload_profile"])
PY
)

SSH_OPTS=(-i "$PEM" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ConnectTimeout=15 -o ServerAliveInterval=30)
ssh_run() { ssh "${SSH_OPTS[@]}" "ubuntu@$IP" "$@"; }

for i in $(seq 1 60); do
  if ssh_run 'echo ssh-ok'; then break; fi
  echo "$TILE waiting for sshd, attempt $i"; sleep 10
done
for i in $(seq 1 90); do
  if ssh_run 'test -f /mnt/nvme/SETUP_DONE'; then break; fi
  echo "$TILE waiting for user-data, attempt $i"
  ssh_run 'tail -2 /var/log/lst-setup.log' || true
  sleep 10
done

# Credentials. The role cannot pass an instance profile, so both sets are
# copied. The SSO set expires in about an hour against a tile that takes most
# of one, which is why this happens immediately before the run and not earlier.
ssh_run 'mkdir -p ~/.aws'
scp "${SSH_OPTS[@]}" ~/.aws/credentials "ubuntu@$IP:~/.aws/credentials"
ssh_run 'chmod 600 ~/.aws/credentials'

ssh_run "git clone $(python3 -c "
import json,sys; print(json.load(open('$MANIFEST'))['config']['paths']['repo_url'])
") /home/ubuntu/landsat-lst-smoke || true"
scp "${SSH_OPTS[@]}" "$(dirname "$0")/run.sh" "$(dirname "$0")/upload.py" \
    "ubuntu@$IP:/home/ubuntu/"
ssh_run 'chmod +x /home/ubuntu/run.sh'

ssh_run "export PATH=\$HOME/.local/bin:\$PATH
  cd /home/ubuntu/landsat-lst-smoke
  setsid nohup uv run /home/ubuntu/upload.py \
    --run-dir /mnt/nvme/run --bucket $BUCKET \
    --prefix $RUNS/$NAME --profile $UPROF \
    > /mnt/nvme/run/upload.log 2>&1 < /dev/null &"

ssh_run "export ART_URI=$ART
  setsid nohup /home/ubuntu/run.sh $TILE $COMMIT \
    > /mnt/nvme/run/nohup.log 2>&1 < /dev/null &"
sleep 8
ssh_run 'pgrep -a -f "run.sh|upload.py" | head -3'
echo "$TILE STARTED  results -> s3://$BUCKET/$RUNS/$NAME"
