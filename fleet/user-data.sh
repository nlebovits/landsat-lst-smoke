#!/bin/bash
# Instance bootstrap: mount the instance store and install what a run needs.
#
# The staging directory must be the instance store. A tile stages 222 to 382
# GiB, `staging.disk_guard` refuses to start without room, and the default
# staging path is the system temp directory, which here is the 150 GB root.
set -x
exec > >(tee -a /var/log/lst-setup.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y git sysstat rsync curl

# The largest disk that is not the root disk. Instance-store naming differs
# between instance families, so this asks by size rather than by name.
ROOT_DISK=$(lsblk -no PKNAME "$(findmnt -no SOURCE /)")
DEV=$(lsblk -dn -b -o NAME,SIZE,TYPE \
      | awk -v r="$ROOT_DISK" '$3=="disk" && $1!=r {print $2, $1}' \
      | sort -rn | head -1 | awk '{print "/dev/"$2}')
echo "root disk $ROOT_DISK, instance store $DEV"
test -n "$DEV" || { echo "NO INSTANCE STORE FOUND"; exit 1; }

mkfs.ext4 -F -E nodiscard "$DEV"
mkdir -p /mnt/nvme
mount -o noatime "$DEV" /mnt/nvme
mkdir -p /mnt/nvme/stage /mnt/nvme/run
chown -R ubuntu:ubuntu /mnt/nvme
df -h /mnt/nvme

su - ubuntu -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'

# The only thing that bounds what a hung run can cost. Terminate-on-shutdown is
# set on the instance, so this halt is a termination. MEASURED tile wall clock
# is 26 to 43 minutes; the previous 120 let a finished box idle for over an
# hour, which cost more than the tile's own compute.
shutdown -h +__DEADLINE__

touch /mnt/nvme/SETUP_DONE
chown ubuntu:ubuntu /mnt/nvme/SETUP_DONE
echo "SETUP COMPLETE"
