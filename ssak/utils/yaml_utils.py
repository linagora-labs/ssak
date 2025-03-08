import re

import hyperpyyaml


def make_yaml_overrides(yaml_file, key_values):
    """
    return a dictionary of overrides to be used with speechbrain
    yaml_file: path to yaml file
    key_values: dict of key values to override
    """
    if yaml_file is None:
        return None
    override = {}
    with open(yaml_file) as f:
        parent = None
        for line in f:
            if line.strip() == "":
                parent = None
            elif line == line.lstrip():
                if ":" in line:
                    parent = line.split(":")[0].strip()
                    if parent in key_values:
                        override[parent] = key_values[parent]
            elif ":" in line:
                child = line.strip().split(":")[0].strip()
                if child in key_values:
                    override[parent] = override.get(parent, {}) | {child: key_values[child]}
    return override


def copy_yaml_fields(from_file, to_file, fields, overrides=""):
    def add_referenced_fields(line):
        if field in fields:
            for ref in re.findall(r"<[^<>]*>", line):
                ref = ref[1:-1]
                fields.append(ref)

    if isinstance(overrides, dict):
        overrides = "\n".join([f"{k}: {v}" for k, v in overrides.items()])
    with open(from_file) as f:
        content = {}
        for source in f, overrides.split("\n"):
            field = None
            for line in source:
                if line == line.lstrip() and line != "" and ":" in line:
                    field = line.split(":")[0].strip()
                    # assert field not in content, f"Duplicate field {field} in {from_file}"
                    content[field] = [line]
                    add_referenced_fields(line)
                elif line.rstrip().startswith("#"):
                    pass
                elif line.strip() != "":
                    assert field is not None, f"Unexpected line {line} in {from_file}"
                    content[field].append(line)
                    add_referenced_fields(line)
                else:
                    field = None
    copied_fields = []
    with open(to_file, "w") as f:
        for (
            field,
            value,
        ) in content.items():
            if field in fields:
                copied_fields.append(field)
                f.write("".join(value) + "\n")
    return copied_fields


def easy_yaml_load(filename, default="PLACEHOLDER"):
    return hyperpyyaml.load_hyperpyyaml(open(filename), overrides=make_yaml_placeholder_overrides(filename, default=default))


def make_yaml_placeholder_overrides(yaml_file, default="PLACEHOLDER"):
    """
    return a dictionary of overrides to be used with speechbrain
    yaml_file: path to yaml file
    key_values: dict of key values to override
    """
    if yaml_file is None:
        return None
    override = {}
    with open(yaml_file) as f:
        parent = None
        for line in f:
            if line == line.lstrip() and line != "" and ":" in line:
                field, value = line.split(":", 1)
                value = value.strip().split()
                if len(value):
                    value = value[0].strip()
                    if value == "!PLACEHOLDER":
                        override[field.strip()] = default
    return override
