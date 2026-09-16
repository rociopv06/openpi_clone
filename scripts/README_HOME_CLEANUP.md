# Home storage monitoring & cleanup

This folder provides two small utilities to help avoid repeatedly filling `$HOME`:

- `home_disk_report.sh` — lists the largest directories and files under `$HOME` and shows free space.
- `home_cleanup.sh` — a safe, idempotent cleanup tool that by default does a dry-run and reports cache/tmp files older than a threshold. Use `--exec` to actually delete.

Suggested usage:

1. Dry-run to see what would be removed:

```bash
bash scripts/home_cleanup.sh --days 30
```

2. If the output looks safe, execute deletion:

```bash
bash scripts/home_cleanup.sh --days 30 --exec
```

3. Check top consumers before and after:

```bash
bash scripts/home_disk_report.sh
```

Scheduling: add a weekly cron job (user crontab) to run the dry-run report and send an email, or run the cleanup with `--exec` only after review.

Example crontab entry (runs report every Sunday 03:00 and appends to a log):

```
0 3 * * 0 /bin/bash /storage/home/hcoda1/3/rvaldes6/r-agarg35-0/scripts/home_disk_report.sh >> /storage/home/hcoda1/3/rvaldes6/r-agarg35-0/scripts/home_disk_report.log 2>&1
```

If you prefer fully automatic deletion, consider a cautious approach: run `home_cleanup.sh --days 60 --exec` monthly and keep backups of important directories.
