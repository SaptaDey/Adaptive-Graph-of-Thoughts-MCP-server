from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
DOC_PATH = Path(__file__).resolve().parents[1] / "docs_src" / "configuration.md"


def dump_section(data, prefix=""):
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"- **{prefix + key}/**")
            lines.extend(dump_section(value, prefix=prefix + key + "."))
        else:
            lines.append(f"- `{prefix + key}`: `{value}`")
    return lines


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    lines = ["# Configuration Reference", ""]
    lines.extend(dump_section(cfg))
    DOC_PATH.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
