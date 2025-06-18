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
    try:
        cfg = yaml.safe_load(CONFIG_PATH.read_text())
        if not cfg:
            raise ValueError("Empty or invalid YAML configuration")
        lines = ["# Configuration Reference", ""]
        lines.extend(dump_section(cfg))
        DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
        DOC_PATH.write_text("\n".join(lines))
        print(f"Documentation generated successfully at {DOC_PATH}")
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML configuration: {e}")
        raise
    except Exception as e:
        print(f"Error: Failed to generate documentation: {e}")
        raise


if __name__ == "__main__":
    main()
