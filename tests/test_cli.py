"""Validate CLI inputs before expensive inference setup."""

import pytest

from clutterpick.cli import build_parser


@pytest.mark.parametrize(
    "command, option",
    [
        ("evaluate", "--limit"),
        ("infer", "--limit"),
        ("benchmark", "--limit"),
        ("train", "--epochs"),
        ("train", "--limit"),
        ("train", "--val-limit"),
    ],
)
@pytest.mark.parametrize("value", ["0", "-1", "1.5", "abc"])
def test_invalid_counts_report_usage_error(command, option, value, capsys):
    with pytest.raises(SystemExit) as error:
        build_parser().parse_args([command, option, value])
    assert error.value.code == 2
    assert "Expected a positive integer" in capsys.readouterr().err


def test_count_defaults_and_explicit_values():
    parser = build_parser()
    assert parser.parse_args(["evaluate"]).limit is None
    assert parser.parse_args(["infer"]).limit == 1
    assert parser.parse_args(["benchmark"]).limit == 100
    args = parser.parse_args(["train", "--epochs", "2", "--limit", "10"])
    assert (args.epochs, args.limit, args.val_limit) == (2, 10, 200)
