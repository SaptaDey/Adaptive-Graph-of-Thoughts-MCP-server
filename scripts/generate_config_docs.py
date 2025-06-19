from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"
DOC_PATH = Path(__file__).resolve().parents[1] / "docs_src" / "configuration.md"


def dump_section(data, prefix=""):
    """
    Recursively generates a Markdown-formatted list representing the structure and contents of a nested configuration dictionary.
    
    Parameters:
        data (dict): The configuration data to document.
        prefix (str): String to prepend to each key, used for representing nested paths.
    
    Returns:
        list[str]: Markdown lines describing the configuration hierarchy and values.
    """
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"- **{prefix + key}/**")
            lines.extend(dump_section(value, prefix=prefix + key + "."))
        else:
            lines.append(f"- `{prefix + key}`: `{value}`")
    return lines


def main() -> None:
    """
    Generate Markdown documentation for the YAML configuration file.
    
    Reads the YAML configuration from a predefined path, validates its content, converts its structure into a Markdown-formatted list, and writes the result to a documentation file. Creates the output directory if it does not exist. Raises exceptions for missing files, invalid YAML, or empty configurations.
    """
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
