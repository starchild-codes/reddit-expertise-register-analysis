# Frozen Collection Failure and Eligibility Rules

Frozen before the fresh corpus has been cleaned or any linguistic outcome is
inspected.

- Target 100--150 eligible AI-related posts per frozen subreddit.
- At least 100 eligible posts: fully usable.
- 80--99: retain and flag as reduced-N; do not replace merely to equalize counts.
- Fewer than 80 because of genuine scarcity or public-page retrieval limits: flag
  for review before outcome analysis. Never replace automatically.
- Do not extend the window for one subreddit. Any future extension must be
  prospective and apply to every frozen community under the same rule.
- Score, comment count, and linguistic characteristics never decide selection,
  retention, usability, replacement, or tier.

For every collection failure the log records the subreddit, attempted window,
discovered and retrievable counts, failure type, and category: content scarcity,
public-page depth, access restriction, or technical error.
