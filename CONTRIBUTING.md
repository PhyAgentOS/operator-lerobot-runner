# Contributing

Thank you for contributing to LeRobot Inference.

## Development

Use Python 3.12 and `uv`:

```bash
uv sync --frozen --extra dev
uv run pytest -q
uv run lerobot --help
uv run lerobot --version
```

Keep changes focused and add tests for behavior changes. Update the
documentation when policy adapters, configuration schemas, conversion output,
or CLI behavior changes.

## Security and generated data

- Report vulnerabilities privately as described in `SECURITY.md`.
- Never commit credentials, private repository URLs, recordings, runtime state,
  machine-specific paths, checkpoint weights, or personal data.
- Do not run tests that command physical hardware without independent safety
  controls and an operator-accessible emergency stop.
- Checkpoint weights referenced in examples are downloaded from public
  Hugging Face repositories at documented revisions.

By submitting a contribution, you agree that it is licensed under Apache
License 2.0.

## README capability summary

Changes to Node entrypoints, Endpoint operations or Tool APIs must update the root
README capability summary in the same change. Follow the [Node README contract](https://github.com/PhyAgentOS/forge/blob/master/CONTRIBUTING.md#node-readme-contract);
count Nodes, Endpoints, operations and deployment Tools separately.
