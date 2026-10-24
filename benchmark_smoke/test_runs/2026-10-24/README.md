# Test run — 2026-10-24

Base revision: `0c7b66027674e5181edf81990ddec24f1ec6c30b` (`origin/main` after fast-forward).
Interpreter: `/home/zomer007/miniforge3/envs/metalyzer/bin/python` (Python 3.14.3, conda).

| Command | Result | Output |
| --- | --- | --- |
| `python -m unittest discover -s tests -p "test_country_regressions.py" -v` | 4 passed | [country_regressions.log](country_regressions.log) |
| `python -m unittest discover -s tests -v` | 143 passed | [full_suite.log](full_suite.log) |

The first full-suite run found 7 failures and 1 error among 143 tests; its output is retained in [full_suite.initial.log](full_suite.initial.log). Seven assertions still expected country names after the ISO 3166-1 alpha-2 output change, and two benchmark fixture checks omitted the now-required `final_country` field. The test expectations and synthetic fixtures were updated; production code and reference labels were not changed in this test run.
