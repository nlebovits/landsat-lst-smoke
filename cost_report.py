#!/usr/bin/env python3
"""Deterministic cost report for a cloud run. No inference, no polling.

Every figure is labelled MEASURED, DERIVED or UNKNOWN, and every derived figure
prints its formula. Runtime comes from the AWS API, never from how long the work
felt: this repo once reported $10.70 for a run that cost about $3, because the
lifetime was inferred from a polling loop rather than read from LaunchTime.

    ./cost_report.py --tag purpose=lst-benchmark --region us-west-2 \
        --profile radiant-earth --requests-per-read 3.1 --shard-scene-reads 605617

Omit --requests-per-read and S3 charges are reported UNKNOWN rather than
guessed. Use measure_s3_requests.py to obtain it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

# us-west-2 Linux on-demand list prices, USD/hr. Pinned because the Pricing API
# needs pricing:GetProducts, which the SSO role used here does not have. Update
# deliberately; a stale rate is a wrong report.
RATES = {
    "c6i.16xlarge": 2.72, "c6i.32xlarge": 5.44, "c6i.8xlarge": 1.36,
    "m6i.4xlarge": 0.768, "m6i.8xlarge": 1.536,
    "r6i.4xlarge": 1.008, "r6i.8xlarge": 2.016, "r6i.16xlarge": 4.032,
}
EBS_GP3_GB_MONTH = 0.08          # USD per GB-month
IPV4_HR = 0.005                  # USD per public IPv4 per hour
S3_GET_PER_1000 = 0.0004         # USD, Standard, requester pays
SEC_PER_MONTH = 730 * 3600


def aws(args, profile, region):
    cmd = ["aws", *args, "--output", "json"]
    if profile:
        cmd += ["--profile", profile]
    if region:
        cmd += ["--region", region]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if out.returncode:
        raise SystemExit(f"aws failed: {out.stderr.strip()[:300]}")
    return json.loads(out.stdout or "{}")


def lifetimes(tag, profile, region):
    """MEASURED: launch and termination straight from the EC2 API."""
    k, _, v = tag.partition("=")
    d = aws(["ec2", "describe-instances", "--filters", f"Name=tag:{k},Values={v}"],
            profile, region)
    rows = []
    for r in d.get("Reservations", []):
        for i in r.get("Instances", []):
            launch = datetime.fromisoformat(i["LaunchTime"].replace("Z", "+00:00"))
            reason = i.get("StateTransitionReason", "")
            term = None
            if "(" in reason:
                ts = reason.split("(", 1)[1].rstrip(")").replace(" GMT", "")
                try:
                    term = datetime.fromisoformat(ts).replace(tzinfo=timezone.utc)
                except ValueError:
                    term = None
            vol = sum(
                bd.get("Ebs", {}).get("VolumeSize", 0)
                for bd in i.get("BlockDeviceMappings", [])
            ) or None
            rows.append({
                "id": i["InstanceId"], "type": i["InstanceType"],
                "state": i["State"]["Name"], "az": i["Placement"]["AvailabilityZone"],
                "ami": i.get("ImageId"), "launch": launch, "term": term,
                "seconds": (term - launch).total_seconds() if term else None,
                "ebs_gb": vol,
            })
    return sorted(rows, key=lambda r: (r["launch"], r["id"]))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", required=True, help="Key=Value used to find the instances")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--profile", default=None)
    p.add_argument("--ebs-gb", type=int, default=None,
                   help="override the EBS size the API reports")
    p.add_argument("--shard-scene-reads", type=int, default=None,
                   help="MEASURED count of shard x scene reads, from the shard plan")
    p.add_argument("--bands", type=int, default=2)
    p.add_argument("--requests-per-read", type=float, default=None,
                   help="MEASURED by measure_s3_requests.py. Omit and S3 is UNKNOWN")
    p.add_argument("--json", type=argparse.FileType("w"), default=None)
    a = p.parse_args()

    rows = lifetimes(a.tag, a.profile, a.region)
    if not rows:
        raise SystemExit(f"no instances matching {a.tag} (terminated ones age out ~1h)")

    print(f"=== MEASURED: instances tagged {a.tag} in {a.region} ===")
    print(f"{'instance':21}{'type':15}{'AMI':23}{'launch':10}{'term':10}{'sec':>7}")
    total_sec, unknown = 0.0, []
    for r in rows:
        t = r["term"].strftime("%H:%M:%S") if r["term"] else "-"
        s = f"{r['seconds']:.0f}" if r["seconds"] is not None else "RUNNING"
        print(f"{r['id']:21}{r['type']:15}{str(r['ami']):23}"
              f"{r['launch'].strftime('%H:%M:%S'):10}{t:10}{s:>7}")
        if r["seconds"] is None:
            unknown.append(r["id"])
        else:
            total_sec += r["seconds"]
    if unknown:
        print(f"  NOTE {len(unknown)} instance(s) still running; excluded from totals")

    by_type: dict[str, float] = {}
    ebs_gb_sec = 0.0
    for r in rows:
        if r["seconds"] is None:
            continue
        by_type[r["type"]] = by_type.get(r["type"], 0.0) + r["seconds"]
        ebs_gb_sec += (a.ebs_gb or r["ebs_gb"] or 0) * r["seconds"]

    print(f"\n=== DERIVED: EC2, from pinned us-west-2 Linux on-demand rates ===")
    ec2 = 0.0
    missing_rate = []
    for t, sec in sorted(by_type.items()):
        rate = RATES.get(t)
        if rate is None:
            missing_rate.append(t)
            print(f"  {t:15} {sec:8.0f}s   RATE NOT PINNED -> excluded")
            continue
        c = sec / 3600 * rate
        ec2 += c
        print(f"  {t:15} {sec:8.0f}s / 3600 x ${rate:<7} = ${c:8.4f}")
    print(f"  {'EC2 subtotal':15} {total_sec:8.0f}s{'':17}= ${ec2:8.4f}")
    if missing_rate:
        print(f"  WARNING rate not pinned for: {', '.join(missing_rate)}")

    ebs = ebs_gb_sec / SEC_PER_MONTH * EBS_GP3_GB_MONTH
    ip4 = total_sec / 3600 * IPV4_HR * len([r for r in rows if r["seconds"] is not None])
    print(f"\n=== DERIVED: storage and address ===")
    print(f"  EBS gp3   {ebs_gb_sec:,.0f} GB-s / {SEC_PER_MONTH:,} x ${EBS_GP3_GB_MONTH} = ${ebs:.4f}")
    print(f"  IPv4      {total_sec/3600:.4f} ih x ${IPV4_HR}{'':16}= ${ip4:.4f}")

    print(f"\n=== S3 (requester pays) ===")
    s3 = None
    if a.shard_scene_reads is None:
        print("  UNKNOWN: --shard-scene-reads not given")
    elif a.requests_per_read is None:
        print(f"  MEASURED   shard-scene reads : {a.shard_scene_reads:,} x {a.bands} bands")
        print( "  UNKNOWN    requests per read : not measured")
        print( "  UNKNOWN    S3 request charge : run measure_s3_requests.py first")
        print( "             Do NOT substitute a guess. At this read pattern the")
        print( "             charge can rival EC2, so a guess can invert a decision.")
    else:
        gets = a.shard_scene_reads * a.bands * a.requests_per_read
        s3 = gets / 1000 * S3_GET_PER_1000
        print(f"  MEASURED   reads x bands x req/read = {a.shard_scene_reads:,} x {a.bands}"
              f" x {a.requests_per_read} = {gets:,.0f} GETs")
        print(f"  DERIVED    {gets:,.0f} / 1000 x ${S3_GET_PER_1000} = ${s3:.4f}")
    print(f"  DERIVED    S3 to EC2 transfer, same region = $0.00")

    known = ec2 + ebs + ip4 + (s3 or 0.0)
    print(f"\n=== TOTAL ===")
    print(f"  known lines                = ${known:.4f}")
    if s3 is None:
        print( "  S3 requests                = UNKNOWN, excluded")
        print(f"  => total is a LOWER BOUND of ${known:.4f}")
    print("\n  Rates are pinned list prices, not billed amounts. Cost Explorer is")
    print("  authoritative; this report is not a bill.")

    if a.json:
        json.dump({
            "measured": {"instances": [
                {**r, "launch": r["launch"].isoformat(),
                 "term": r["term"].isoformat() if r["term"] else None} for r in rows],
                "total_instance_seconds": total_sec,
                "shard_scene_reads": a.shard_scene_reads},
            "derived": {"ec2_usd": ec2, "ebs_usd": ebs, "ipv4_usd": ip4, "s3_usd": s3},
            "unknown": [] if s3 is not None else ["s3_request_charge"],
            "rates": {"ec2": RATES, "ebs_gb_month": EBS_GP3_GB_MONTH,
                      "ipv4_hr": IPV4_HR, "s3_get_per_1000": S3_GET_PER_1000},
        }, a.json, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
