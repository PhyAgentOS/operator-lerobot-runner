## Summary

-

## Validation

- [ ] `uv sync --frozen --extra dev` succeeds.
- [ ] `uv run pytest -q` passes.
- [ ] CLI help and version checks pass.
- [ ] No secrets, recordings, machine paths, checkpoint weights, or private repository URLs are included.
- [ ] License notices are updated when dependencies or bundled resources change.

## Deployment and safety

- [ ] The change does not weaken the trusted-network-only guidance for Dora dataflows.
- [ ] Any physical robot validation used independent safety controls and an operator-accessible emergency stop.

Describe deployment, network exposure, and physical validation performed:
