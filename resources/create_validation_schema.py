import json
import re


def map_type(type_str):
    """Map the type from the Markdown table to a JSON Schema type."""
    type_str = type_str.strip().lower()
    if type_str == "integer":
        return "integer"
    elif type_str == "double":
        return "number"
    elif type_str == "string":
        return "string"
    elif type_str == "boolean":
        return "boolean"
    return "string"


def convert_node_to_schema(node):
    """
    Recursively converts the nested dictionary (from the Markdown table)
    into a JSON Schema 'object' with properties and required fields.
    """
    properties = {}
    required = []
    for key, value in node.items():
        # If this dictionary has a "type" key, it represents a parameter row.
        if isinstance(value, dict) and "type" in value:
            json_type = map_type(value["type"])
            prop_schema = {"type": json_type}
            default_value = value.get("default")
            if default_value != "":
                try:
                    if json_type == "integer":
                        prop_schema["default"] = int(default_value)
                    elif json_type == "number":
                        prop_schema["default"] = float(default_value)
                    else:
                        prop_schema["default"] = default_value
                except Exception:
                    prop_schema["default"] = default_value
            comment = value.get("comment")
            if comment:
                prop_schema["description"] = comment
            properties[key] = prop_schema
            # Add key to required list if marked compulsory
            compulsory = value.get("compulsory")
            if compulsory is True or (
                isinstance(compulsory, str) and compulsory.lower() == "yes"
            ):
                required.append(key)
        else:
            # Otherwise, assume it's a nested group/chapter and recurse.
            group_schema = convert_node_to_schema(value)
            properties[key] = group_schema
    schema_obj = {"type": "object", "properties": properties}
    if required:
        schema_obj["required"] = required
    return schema_obj


def md_table_to_json_schema_template(md_filepath, json_schema_filepath):
    """
    Reads a Markdown table file and converts it into a JSON Schema file.
    Chapters (lines starting with [) become groups in the schema and parameter
    rows are mapped to schema properties.
    """
    nested_dict = {}
    current_chapter = None
    current_subchapter = None
    current_subsubchapter = None

    with open(md_filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Remove empty lines and strip whitespace.
    lines = [line.strip() for line in lines if line.strip()]
    if len(lines) < 2:
        raise ValueError(
            "The Markdown file does not contain enough lines to be a table."
        )
    # Skip header and separator lines.
    content_lines = lines[2:]

    for line in content_lines:
        # Skip separator lines.
        if re.match(r"^\|[-\s|]+$", line):
            continue

        # Remove the leading and trailing pipe characters.
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]

        # Split cells by pipe.
        cells = [cell.strip() for cell in line.split("|")]
        if not cells or not cells[0]:
            continue

        variable = cells[0]

        # Check for chapter markers (lines starting with '[').
        if variable.startswith("["):
            level = 0
            for ch in variable:
                if ch == "[":
                    level += 1
                else:
                    break
            title = variable.strip("[]").strip()
            if level == 1:
                current_chapter = title
                nested_dict.setdefault(current_chapter, {})
                current_subchapter = None
                current_subsubchapter = None
            elif level == 2:
                if current_chapter is None:
                    raise ValueError(
                        "Found a subchapter before any chapter is defined."
                    )
                current_subchapter = title
                nested_dict[current_chapter].setdefault(current_subchapter, {})
                current_subsubchapter = None
            elif level == 3:
                if current_chapter is None or current_subchapter is None:
                    raise ValueError(
                        "Found a sub-subchapter without a parent chapter and \
                           subchapter."
                    )
                current_subsubchapter = title
                nested_dict[current_chapter][current_subchapter].setdefault(
                    current_subsubchapter, {}
                )
            continue

        # Otherwise, treat the row as a parameter row.
        # Expected columns: Variable, Type, Compulsory, Default Value, Comment
        param_type = cells[1] if len(cells) > 1 else ""
        compulsory = cells[2] if len(cells) > 2 else ""
        default_value = cells[3] if len(cells) > 3 else ""
        comment = cells[4] if len(cells) > 4 else ""

        # Normalize the compulsory flag.
        if compulsory.lower() == "yes":
            compulsory_val = True
        elif compulsory.lower() == "no":
            compulsory_val = False
        else:
            compulsory_val = compulsory

        param_dict = {
            "type": param_type,
            "compulsory": compulsory_val,
            "default": default_value,
            "comment": comment,
        }

        # Insert the parameter into the correct hierarchy.
        if current_chapter is None:
            nested_dict[variable] = param_dict
        elif current_subchapter is None:
            nested_dict[current_chapter][variable] = param_dict
        elif current_subsubchapter is None:
            nested_dict[current_chapter][current_subchapter][variable] = param_dict
        else:
            nested_dict[current_chapter][current_subchapter][current_subsubchapter][
                variable
            ] = param_dict

    # Convert the nested dictionary into a JSON Schema.
    # The top-level schema is an object whose properties are the top-level
    # groups.
    json_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Configuration Schema",
        "type": "object",
        "properties": convert_node_to_schema(nested_dict)["properties"],
    }
    # Optionally, you could add a "required" list at the top level if needed.

    with open(json_schema_filepath, "w", encoding="utf-8") as f:
        json.dump(json_schema, f, indent=2)


md_file = r"./sedtrails_config.md"

# Save to a JSON file
output_file = r"./config_schema.json"
json_schema = md_table_to_json_schema_template(md_file, output_file)


print(f"JSON Schema has been saved to {output_file}")
