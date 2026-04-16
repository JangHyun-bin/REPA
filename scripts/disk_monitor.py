"""Polls disk free space, logs to data/disk_log.csv, exits non-zero if below threshold."""
import argparse, csv, datetime, shutil, sys, time

def main(args):
    with open(args.log, "a") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["timestamp", "free_gb", "used_gb", "alert"])
        while True:
            total, used, free = shutil.disk_usage(args.path)
            free_gb = free / 1e9
            used_gb = used / 1e9
            alert = free_gb < args.min_gb
            w.writerow([datetime.datetime.now().isoformat(),
                        f"{free_gb:.2f}", f"{used_gb:.2f}", int(alert)])
            f.flush()
            if alert:
                print(f"⚠ DISK ALERT: free {free_gb:.1f} GB < threshold {args.min_gb} GB",
                      file=sys.stderr)
                if args.exit_on_alert:
                    sys.exit(1)
            if args.once:
                break
            time.sleep(args.interval)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--path", default=".")
    p.add_argument("--log", default="data/disk_log.csv")
    p.add_argument("--min_gb", type=float, default=15.0)
    p.add_argument("--interval", type=int, default=60)
    p.add_argument("--once", action="store_true")
    p.add_argument("--exit_on_alert", action="store_true")
    main(p.parse_args())
