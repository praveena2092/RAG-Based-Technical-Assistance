"""Step 1a - Ingest course material by cloning the public TDS GitHub repo."""
from __future__ import annotations

import shutil
import subprocess

from tds_rag import config

EXTENSIONS = {
    "markdown": (".md",),
    "images": (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg"),
    "data": (".csv", ".xlsx", ".xls", ".json"),
    "html": (".html",),
    "python": (".py",),
}
EXPORT_CATEGORIES = ("images", "python", "html")


def clone_repo() -> None:
    if config.COURSE_REPO_DIR.exists():
        print(f"Removing existing folder: {config.COURSE_REPO_DIR}")
        shutil.rmtree(config.COURSE_REPO_DIR)
    config.COURSE_REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {config.COURSE_REPO_URL}")
    subprocess.run(
        ["git", "clone", "--depth", "1", config.COURSE_REPO_URL, str(config.COURSE_REPO_DIR)],
        check=True,
    )


def collect_files() -> dict[str, list]:
    file_map = {key: [] for key in EXTENSIONS}
    for path in config.COURSE_REPO_DIR.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        for key, exts in EXTENSIONS.items():
            if path.name.lower().endswith(exts):
                file_map[key].append(path)
    return file_map


def save_summary(file_map: dict[str, list]) -> None:
    with open(config.COURSE_SUMMARY_FILE, "w", encoding="utf-8") as f:
        for category, paths in file_map.items():
            f.write(f"\n{category.upper()} FILES:\n")
            for path in paths:
                f.write(f"{path}\n")


def export_files(file_map: dict[str, list]) -> None:
    for category in EXPORT_CATEGORIES:
        out_dir = config.COURSE_ASSETS_DIR / category
        out_dir.mkdir(parents=True, exist_ok=True)
        for src in file_map.get(category, []):
            try:
                shutil.copy(src, out_dir)
            except Exception as e:  # noqa: BLE001
                print(f"Failed to copy {src}: {e}")


def main() -> None:
    clone_repo()
    file_map = collect_files()
    save_summary(file_map)
    export_files(file_map)
    print(f"Done. Markdown files found: {len(file_map['markdown'])}")
    print(f"Repo: {config.COURSE_REPO_DIR}")


if __name__ == "__main__":
    main()
