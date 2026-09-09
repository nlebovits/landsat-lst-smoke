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

# Fallback only. Live rates come from AWS's public price list, which needs no
# credentials; the Pricing API needs pricing:GetProducts, which this SSO role
# lacks. A pinned rate that has gone stale is a wrong report, so these are used
# only when the fetch fails, and the report says which source it used.
RATES_FALLBACK = {
    "c6i.16xlarge": 2.72,
    "c6i.32xlarge": 5.44,
    "c6i.8xlarge": 1.36,
    "m6i.4xlarge": 0.768,
    "m6i.8xlarge": 1.536,
    "r6i.4xlarge": 1.008,
    "r6i.8xlarge": 2.016,
    "r6i.16xlarge": 4.032,
}
PRICE_URL = (
    "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/"
    "current/ec2-ondemand-without-sec-sel/{loc}/Linux/index.json"
)
LOCATIONS = {
    "us-west-2": "US West (Oregon)",
    "us-east-1": "US East (N. Virginia)",
    "eu-central-1": "EU (Frankfurt)",
    "eu-west-1": "EU (Ireland)",
}


def fetch_rates(region):
    """VERIFIED rates from the public price list. Returns (rates, source)."""
    import gzip
    import urllib.parse
    import urllib.request

    loc = LOCATIONS.get(region)
    if not loc:
        return RATES_FALLBACK, f"pinned fallback (no location map for {region})"
    url = PRICE_URL.format(loc=urllib.parse.quote(loc))
    try:
        with urllib.request.urlopen(url, timeout=45) as r:
            raw = r.read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        d = json.loads(raw)
        out = {}
        for items in d.get("regions", {}).values():
            for v in items.values():
                t = v.get("Instance Type")
                if t and v.get("price"):
                    out[t] = float(v["price"])
        if out:
            return out, "VERIFIED from AWS public price list"
    except Exception as exc:
        return RATES_FALLBACK, f"pinned fallback (fetch failed: {type(exc).__name__})"
    return RATES_FALLBACK, "pinned fallback (empty response)"


EBS_GP3_GB_MONTH = 0.08  # USD per GB-month
IPV4_HR = 0.005  # USD per public IPv4 per hour
S3_GET_PER_1000 = 0.0004  # USD, Standard, requester pays
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
    d = aws(
        ["ec2", "describe-instances", "--filters", f"Name=tag:{k},Values={v}"],
        profile,
        region,
    )
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
            vol = (
                sum(
                    bd.get("Ebs", {}).get("VolumeSize", 0)
                    for bd in i.get("BlockDeviceMappings", [])
                )
                or None
            )
            rows.append(
                {
                    "id": i["InstanceId"],
                    "type": i["InstanceType"],
                    "state": i["State"]["Name"],
                    "az": i["Placement"]["AvailabilityZone"],
                    "ami": i.get("ImageId"),
                    "launch": launch,
                    "term": term,
                    "seconds": (term - launch).total_seconds() if term else None,
                    "ebs_gb": vol,
                }
            )
    return sorted(rows, key=lambda r: (r["launch"], r["id"]))


# The CLI entry point: argument parsing, then one branch per cost line, then
# the report. Splitting it would scatter the arithmetic this script exists to
# make auditable in one place.
def main() -> int:  # noqa: C901
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--tag", required=True, help="Key=Value used to find the instances")
    p.add_argument("--region", default="us-west-2")
    p.add_argument("--profile", default=None)
    p.add_argument(
        "--ebs-gb", type=int, default=None, help="override the EBS size the API reports"
    )
    p.add_argument(
        "--shard-scene-reads",
        type=int,
        default=None,
        help="MEASURED count of shard x scene reads, from the shard plan",
    )
    p.add_argument("--bands", type=int, default=2)
    p.add_argument(
        "--requests-per-read",
        type=float,
        default=None,
        help="MEASURED by measure_s3_requests.py. Omit and S3 is UNKNOWN",
    )
    p.add_argument(
        "--s3-get-requests",
        type=int,
        default=None,
        help="MEASURED total GETs, from the staging.json a staged run writes. "
        "Staging fetches each object once, so the run counts its own "
        "requests and nothing has to be derived from a sample. Takes "
        "precedence over --shard-scene-reads",
    )
    p.add_argument(
        "--recorded",
        action="append",
        default=[],
        metavar="TYPE:COUNT:SECONDS",
        help="price a fleet from recorded lifetimes instead of the "
        "API, for a run whose instances have aged out of "
        "describe-instances. Repeatable",
    )
    # FileType is deprecated from 3.14. This repository pins 3.12, and
    # swapping it changes how the CLI reports an unwritable path.
    p.add_argument(
        "--json",
        type=argparse.FileType("w"),  # ty: ignore[deprecated]
        default=None,
    )
    a = p.parse_args()

    RATES, rate_src = fetch_rates(a.region)

    recorded = []
    for spec in a.recorded:
        try:
            t, count, sec = spec.split(":")
            recorded.append((t, int(count), float(sec)))
        except ValueError:
            raise SystemExit(f"--recorded wants TYPE:COUNT:SECONDS, got {spec!r}")

    rows = lifetimes(a.tag, a.profile, a.region) if not recorded else []
    if not rows and not recorded:
        print("=== EC2: UNAVAILABLE ===")
        print(f"  no instances matching {a.tag} in {a.region}.")
        print("  Terminated instances age out of describe-instances after about")
        print("  an hour. Pass --recorded TYPE:COUNT:SECONDS to price a past run")
        print("  from its recorded lifetimes, or run this straight after teardown.")
        print("  Every EC2 line below is omitted, not zero.")

    if rows:
        print(f"=== MEASURED: instances tagged {a.tag} in {a.region} ===")
        print(f"{'instance':21}{'type':15}{'AMI':23}{'launch':10}{'term':10}{'sec':>7}")
    total_sec, unknown = 0.0, []
    for r in rows:
        t = r["term"].strftime("%H:%M:%S") if r["term"] else "-"
        s = f"{r['seconds']:.0f}" if r["seconds"] is not None else "RUNNING"
        print(
            f"{r['id']:21}{r['type']:15}{str(r['ami']):23}"
            f"{r['launch'].strftime('%H:%M:%S'):10}{t:10}{s:>7}"
        )
        if r["seconds"] is None:
            unknown.append(r["id"])
        else:
            total_sec += r["seconds"]
    if unknown:
        print(f"  NOTE {len(unknown)} instance(s) still running; excluded from totals")

    by_type: dict[str, float] = {}
    ebs_gb_sec = 0.0
    n_instances = len([r for r in rows if r["seconds"] is not None])
    for r in rows:
        if r["seconds"] is None:
            continue
        by_type[r["type"]] = by_type.get(r["type"], 0.0) + r["seconds"]
        ebs_gb_sec += (a.ebs_gb or r["ebs_gb"] or 0) * r["seconds"]

    if recorded:
        print("=== MEASURED: recorded lifetimes, not from the API ===")
        for t, count, sec in recorded:
            print(f"  {t:15} {count} x {sec:.0f}s = {count * sec:.0f}s")
            by_type[t] = by_type.get(t, 0.0) + count * sec
            ebs_gb_sec += (a.ebs_gb or 0) * count * sec
            total_sec += count * sec
            n_instances += count

    have_ec2 = bool(by_type)
    print("\n=== DERIVED: EC2 ===" if have_ec2 else "\n=== DERIVED: EC2, OMITTED ===")
    print(f"  rate source: {rate_src}")
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
    ip4 = total_sec / 3600 * IPV4_HR * n_instances
    if have_ec2:
        print("\n=== DERIVED: storage and address ===")
        print(
            f"  EBS gp3   {ebs_gb_sec:,.0f} GB-s / {SEC_PER_MONTH:,} x ${EBS_GP3_GB_MONTH} = ${ebs:.4f}"
        )
        print(f"  IPv4      {total_sec / 3600:.4f} ih x ${IPV4_HR}{'':16}= ${ip4:.4f}")

    print("\n=== S3 (requester pays) ===")
    s3 = None
    if a.s3_get_requests is not None:
        # A staged run fetches each object once and counts every attempt, so
        # this is the wire total rather than reads x bands x a sampled rate.
        # Nothing here is estimated, which is why it comes first.
        s3 = a.s3_get_requests / 1000 * S3_GET_PER_1000
        print(f"  MEASURED   GETs counted on the wire : {a.s3_get_requests:,}")
        print(
            f"  DERIVED    {a.s3_get_requests:,} / 1000 x ${S3_GET_PER_1000} = ${s3:.4f}"
        )
    elif a.shard_scene_reads is None:
        print("  UNKNOWN: --shard-scene-reads not given")
    elif a.requests_per_read is None:
        print(
            f"  MEASURED   shard-scene reads : {a.shard_scene_reads:,} x {a.bands} bands"
        )
        print("  UNKNOWN    requests per read : not measured")
        print("  UNKNOWN    S3 request charge : run measure_s3_requests.py first")
        print("             Do NOT substitute a guess. At this read pattern the")
        print("             charge can rival EC2, so a guess can invert a decision.")
    else:
        gets = a.shard_scene_reads * a.bands * a.requests_per_read
        s3 = gets / 1000 * S3_GET_PER_1000
        print(
            f"  MEASURED   reads x bands x req/read = {a.shard_scene_reads:,} x {a.bands}"
            f" x {a.requests_per_read} = {gets:,.0f} GETs"
        )
        print(f"  DERIVED    {gets:,.0f} / 1000 x ${S3_GET_PER_1000} = ${s3:.4f}")
    print("  DERIVED    S3 to EC2 transfer, same region = $0.00")

    known = ec2 + ebs + ip4 + (s3 or 0.0)
    print("\n=== TOTAL ===")
    print(f"  known lines                = ${known:.4f}")
    if not have_ec2:
        print("  EC2, EBS, IPv4             = UNAVAILABLE, excluded")
        print("  => this total covers S3 requests only")
    if s3 is None:
        print("  S3 requests                = UNKNOWN, excluded")
        print(f"  => total is a LOWER BOUND of ${known:.4f}")
    print(f"\n  EC2 rate: {rate_src}.")
    print("  EBS, IPv4 and S3 rates are published values, not fetched.")
    print("  These are list prices, not billed amounts. Cost Explorer is")
    print("  authoritative; this report is not a bill.")

    if a.json:
        json.dump(
            {
                "measured": {
                    "instances": [
                        {
                            **r,
                            "launch": r["launch"].isoformat(),
                            "term": r["term"].isoformat() if r["term"] else None,
                        }
                        for r in rows
                    ],
                    "total_instance_seconds": total_sec,
                    "shard_scene_reads": a.shard_scene_reads,
                    "s3_get_requests": a.s3_get_requests,
                },
                "derived": {
                    "ec2_usd": ec2,
                    "ebs_usd": ebs,
                    "ipv4_usd": ip4,
                    "s3_usd": s3,
                },
                "unknown": [] if s3 is not None else ["s3_request_charge"],
                "rates": {
                    "ec2_source": rate_src,
                    "ec2": RATES,
                    "ebs_gb_month": EBS_GP3_GB_MONTH,
                    "ipv4_hr": IPV4_HR,
                    "s3_get_per_1000": S3_GET_PER_1000,
                },
            },
            a.json,
            indent=2,
            default=str,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
