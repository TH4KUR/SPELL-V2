"""SPELL-RQ2 data-selection framework (PROTOCOL §3.19–3.21, §5 item 12).

Roster (risk order, HANDOFF §4): dnsmos → kmeans → dsir → proxy_rank
(lossrank/el2n/anti) → less. Each selector is a pure function over a committed
score table + the shared SelectionContext; manifests are written and validated
exclusively through :mod:`selection.base`.
"""
