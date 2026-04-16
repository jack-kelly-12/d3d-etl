import unittest

import pandas as pd

from processors.map_ncaa_to_cube import _parse_jersey_from_name, _resolve


class TestJerseyInName(unittest.TestCase):
    def test_parse_hash_dash_name(self) -> None:
        self.assertEqual(_parse_jersey_from_name("# 12 - Jane Smith"), (12, "Jane Smith"))

    def test_parse_no_dot_dash(self) -> None:
        self.assertEqual(_parse_jersey_from_name("No. 7 – Alex Lee"), (7, "Alex Lee"))

    def test_parse_digits_only(self) -> None:
        self.assertEqual(_parse_jersey_from_name("42"), (42, ""))

    def test_parse_plain_name(self) -> None:
        self.assertEqual(_parse_jersey_from_name("Sam Jones"), (None, "Sam Jones"))

    def test_resolve_maps_jersey_from_name_when_number_missing(self) -> None:
        key = ("testslug", 2026)
        by_name = {key: {"Jane Smith": "pid_js"}}
        by_last: dict = {key: {}}
        by_initlast: dict = {key: {}}
        by_number = {key: {12: "pid_js"}}

        pid = _resolve(
            "# 12 - Jane Smith",
            pd.NA,
            key,
            by_name,
            by_last,
            by_initlast,
            by_number,
        )
        self.assertEqual(pid, "pid_js")

    def test_resolve_column_number_wins_over_wrong_prefix_in_name(self) -> None:
        key = ("testslug", 2026)
        by_number = {key: {5: "pid5", 12: "pid12"}}
        by_name = {key: {"Other Guy": "pid5"}}
        by_last: dict = {key: {}}
        by_initlast: dict = {key: {}}

        pid = _resolve(
            "# 12 - Other Guy",
            5,
            key,
            by_name,
            by_last,
            by_initlast,
            by_number,
        )
        self.assertEqual(pid, "pid5")

    def test_resolve_stripped_name_for_exact_match_after_number_miss(self) -> None:
        key = ("testslug", 2026)
        by_number = {key: {99: "pid99"}}
        by_name = {key: {"Pat Brown": "pid_pb"}}
        by_last = {key: {"brown": ["pid_pb"]}}
        by_initlast: dict = {key: {}}

        pid = _resolve(
            "# 12 - Pat Brown",
            pd.NA,
            key,
            by_name,
            by_last,
            by_initlast,
            by_number,
        )
        self.assertEqual(pid, "pid_pb")


if __name__ == "__main__":
    unittest.main()
