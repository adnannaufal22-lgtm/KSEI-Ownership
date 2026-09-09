import unittest

import pandas as pd

from monthly_changes import (
    build_monthly_change_detail,
    owner_change_summary,
    stock_change_summary,
)


class MonthlyChangeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pd.DataFrame(
            [
                ("2026-07-31", "AAA", "Alpha", "Owner A", 100, 10.0),
                ("2026-07-31", "AAA", "Alpha", "Owner B", 50, 5.0),
                ("2026-07-31", "BBB", "Beta", "Owner A", 40, 4.0),
                ("2026-08-31", "AAA", "Alpha", "Owner A", 130, 13.0),
                ("2026-08-31", "AAA", "Alpha", "Owner C", 20, 2.0),
                ("2026-08-31", "BBB", "Beta", "Owner A", 25, 2.5),
            ],
            columns=[
                "date",
                "ticker",
                "security_name",
                "investor_name",
                "ownership_units",
                "ownership_pct",
            ],
        )

    def test_detail_handles_continuing_new_and_removed_reports(self) -> None:
        detail = build_monthly_change_detail(self.data)
        self.assertEqual(len(detail), 4)
        indexed = detail.set_index(["stock", "owner"])
        self.assertEqual(indexed.loc[("AAA", "Owner A"), "change_shares"], 30)
        self.assertEqual(indexed.loc[("AAA", "Owner B"), "observation_status"], "No longer reported")
        self.assertEqual(indexed.loc[("AAA", "Owner C"), "observation_status"], "Newly reported")
        self.assertEqual(indexed.loc[("BBB", "Owner A"), "change_percentage"], -1.5)

    def test_stock_summary_ranks_by_total_absolute_activity(self) -> None:
        summary = stock_change_summary(build_monthly_change_detail(self.data))
        aaa = summary.set_index("stock").loc["AAA"]
        self.assertEqual(aaa["previous_shares"], 150)
        self.assertEqual(aaa["current_shares"], 150)
        self.assertEqual(aaa["net_change"], 0)
        self.assertEqual(aaa["absolute_change"], 100)
        self.assertEqual(aaa["changing_holders"], 3)

    def test_owner_summary_counts_increases_and_decreases(self) -> None:
        summary = owner_change_summary(build_monthly_change_detail(self.data))
        owner_a = summary.set_index("owner").loc["Owner A"]
        self.assertEqual(owner_a["stocks_increased"], 1)
        self.assertEqual(owner_a["stocks_decreased"], 1)
        self.assertEqual(owner_a["stocks_changed"], 2)
        self.assertEqual(owner_a["absolute_change"], 45)


if __name__ == "__main__":
    unittest.main()
