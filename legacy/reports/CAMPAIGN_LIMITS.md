# Campaign limits and provider quota

The previous 750,000-token ceiling was stored in Quant Lab's local SQLite campaign, not reported by MiMo and not requested as a spending ceiling by the user. It must not be described as the user's monthly allowance. The provider account quota has not been inspected.

Following the user's 1 October clarification, new campaigns have no local aggregate cap by default. CLI `plan` omits the cap unless `--token-budget N` is supplied; the dashboard's optional cap is blank by default. SQLite and JSON use `null` for no cap, rather than a fabricated large number. Autonomous reports use `remaining_tokens=null` for these campaigns.

The original campaign `2b22519842f6` remains stopped with its historical 750,000 cap and recorded usage: 686,218 known reported tokens, 746,644 charged/reserved across 26 requests. Its failed scopes remain failed. Continued research uses a separate uncapped campaign with the same frozen plan and shared durable evidence, preserving all prior accounting and global date exposure.

No cap does not erase usage. Actual provider usage remains charged; unknown usage retains its reservation. Explicit task budgets remain atomic and immutable. Request completion bounds, role/tool turn limits, process leases, cooperative runtime deadlines, user stops and finite failure recovery continue to protect execution from stalls. They describe execution controls, not an account quota.

Verification: 29 focused tests passed, including real local HTTP requests for omitted/null/explicit dashboard caps, uncapped usage persistence, task reservation concurrency, finite-cap preservation and failure stops. The complete regression result and current real run are recorded separately.

Complete regression: 346 passed, one platform skip, 137.25 seconds. Continued real campaign `87f69ac0728e` is uncapped and was launched through the fresh-process research runtime. See `results/desk/verification/uncapped-campaign/verification.json` and `real-run.json` for exact source and launch evidence. No completed strategy, reviewed company outcome or promotion is inferred from a running process.
