# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in MalVec, please report it responsibly.

**Do not open a public issue for security vulnerabilities.**

Instead, please email the maintainers directly or use GitHub's [private vulnerability reporting](https://github.com/mwill20/MalVec/security/advisories/new) feature.

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We will acknowledge receipt within 48 hours and provide a timeline for resolution.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Security Model

MalVec processes potentially hostile files (malware samples). The security architecture is designed around five non-negotiable invariants:

1. **Malware never executes** — all analysis is static
2. **File paths never in output** — SHA256 hashes used instead
3. **All inputs validated at boundaries** — size, format, magic bytes
4. **Sandboxing enforced** — 30-second timeout, 512MB memory cap, no network
5. **Fail safely** — errors produce "needs review," never a false clean

For the full threat model and defense-in-depth architecture, see [docs/SECURITY.md](docs/SECURITY.md).

## Dependencies

MalVec's dependencies are monitored for known vulnerabilities via:
- `pip-audit` in CI pipeline
- Dependabot automated PRs for dependency updates
