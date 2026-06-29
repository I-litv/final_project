import subprocess
import sys

from services.data import PROJECT_DIR


def run_training_with_live_logs(
    render_log,
    max_listings,
    start_page,
    min_delay,
    max_delay,
    allow_partial_scrape,
    skip_scrape=False,
):
    """Run training and stream terminal output into the Streamlit page."""
    command = [
        sys.executable,
        "-u",
        "train_model.py",
        "used_cars.csv",
    ]
    if skip_scrape:
        command.append("--skip-scrape")
        log_lines = [
            "$ " + " ".join(command),
            "Training from the current CSV. No scraping will run.",
            "Existing CSV will not be cleared or modified by the scraper.",
            "Starting model training...",
        ]
    else:
        command.extend(
            [
                "--max-listings",
                str(int(max_listings)),
                "--start-page",
                str(int(start_page)),
                "--scrape-min-delay",
                str(float(min_delay)),
                "--scrape-max-delay",
                str(float(max_delay)),
            ]
        )
        if allow_partial_scrape:
            command.append("--allow-partial-scrape")

        log_lines = [
            "$ " + " ".join(command),
            "Existing CSV will be cleared before scraping fresh listings.",
            f"Scraper target: up to {int(max_listings):,} listings.",
            f"Starting AUTO.RIA page: {int(start_page)}.",
            f"Delay between pages: {float(min_delay):.1f}-{float(max_delay):.1f} seconds.",
            "Starting model training...",
        ]
    render_log(log_lines)

    process = subprocess.Popen(
        command,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        for line in process.stdout:
            log_lines.append(line.rstrip())
            render_log(log_lines)

    return_code = process.wait()
    if return_code == 0:
        log_lines.append("Training process finished successfully.")
    else:
        log_lines.append(f"Training process failed with exit code {return_code}.")

    render_log(log_lines)
    return return_code
