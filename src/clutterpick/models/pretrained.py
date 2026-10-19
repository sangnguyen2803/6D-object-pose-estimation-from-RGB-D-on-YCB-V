"""Read published YAML namespaces without arbitrary object construction."""

from pathlib import Path, PurePosixPath

import yaml


class PublishedConfigLoader(yaml.SafeLoader):
    pass


PublishedConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object:argparse.Namespace",
    lambda loader, node: loader.construct_mapping(node, deep=True),
)
PublishedConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node, deep=True)),
)
PublishedConfigLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    lambda loader, node: str(
        PurePosixPath(*loader.construct_sequence(node, deep=True))
    ),
)


def read_published_config(path):
    return yaml.load(
        Path(path).read_text(encoding="utf-8"), Loader=PublishedConfigLoader
    )
