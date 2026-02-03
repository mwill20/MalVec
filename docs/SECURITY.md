# MalVec Security Architecture

This document describes the security architecture of MalVec, a malware classification system using vector similarity.

## Defense-in-Depth Layers

MalVec implements multiple security layers to protect against malicious samples:

### Layer 1: Input Validation

Before any processing, all inputs are validated:

- **File size limits**: Maximum 50MB (configurable)
- **Magic byte verification**: MZ header required for PE files
- **Format validation**: PE/ELF structure verification via LIEF
- **Path validation**: Prevents path traversal attacks

```python
from malvec.validator import InputValidator

file_path = InputValidator.validate("/path/to/sample.exe")
```

### Layer 2: Process Isolation

Feature extraction runs in a separate subprocess:

- **Crash isolation**: LIEF/C++ crashes don't affect main process
- **Clean process state**: Each extraction starts fresh
- **Exit code monitoring**: Abnormal exits are detected

```python
from malvec.isolation import run_isolated

result = run_isolated(extract_features, file_path, timeout=30)
```

### Layer 3: Sandboxing

The sandbox provides resource limits and timeout enforcement:

- **Timeout enforcement**: Default 30 seconds, configurable
- **Memory limits**: Default 512MB (enforced on Linux/macOS)
- **File size limits**: Prevents processing of oversized files
- **Temp directory isolation**: Sandboxed operations use isolated temp space

```python
from malvec.sandbox import SandboxContext, SandboxConfig

config = SandboxConfig(timeout=30, max_memory=512*1024*1024)
with SandboxContext(config) as sandbox:
    features = sandbox.run(extract_features, file_path)
```

### Layer 4: Audit Logging

All security-relevant events are logged in structured JSON:

- **Classification results**: Predictions, confidence, timing
- **Validation failures**: Rejected files with reasons
- **Sandbox violations**: Timeouts, crashes, resource exhaustion
- **File identification**: SHA256 hashes (not filenames by default)

```python
from malvec.audit import AuditLogger

audit = AuditLogger()
audit.log_classification(file_path, "MALWARE", 0.95, 1.23)
```

### Layer 5: Fail-Safe Defaults

The system defaults to safe behavior:

- **Sandbox enabled by default**: Feature extraction is sandboxed
- **Conservative thresholds**: Low-confidence results flagged for review
- **Error handling**: Failures result in safe termination
- **No execution**: Static analysis only, malware never runs

## Security Invariants

**CRITICAL - These must NEVER be violated:**

1. **Malware NEVER executes** - Static analysis only, no dynamic execution
2. **File paths NOT in logs** - Use SHA256 hashes for identification
3. **All inputs validated** - Every boundary validates its inputs
4. **Sandboxing enforced** - Timeouts and memory limits always active
5. **Fail safely** - Errors trigger safe termination, not bypass

## Configuration

### Sandbox Configuration

```python
from malvec.sandbox import SandboxConfig

config = SandboxConfig(
    timeout=30,              # Max execution time (seconds)
    max_memory=512*1024*1024, # Max memory (512MB)
    max_filesize=50*1024*1024, # Max file size (50MB)
    allow_network=False,     # Network isolation (advisory)
)
```

### Audit Configuration

```python
from malvec.audit import AuditConfig

config = AuditConfig(
    log_dir=Path("/var/log/malvec"),
    log_filename="audit.log",
    include_file_names=False,  # Privacy: hash-only by default
    max_log_size=10*1024*1024, # 10MB before rotation
    backup_count=5,            # Keep 5 backup logs
)
```

## Threat Model

### In Scope

The following threats are considered and mitigated:

| Threat | Mitigation |
|--------|------------|
| Malicious PE exploiting parser | Process isolation, LIEF sandboxing |
| Memory exhaustion attacks | Memory limits (512MB default) |
| Infinite loop/hang attacks | Timeout enforcement (30s default) |
| Path traversal attempts | Input validation, path canonicalization |
| Large file DoS | File size limits (50MB default) |
| Process crashes | Subprocess isolation, crash detection |

### Out of Scope

The following are not addressed by MalVec's security model:

- Physical access to analysis machine
- Kernel-level exploits (use VM isolation)
- Side-channel attacks
- Social engineering
- Network-based attacks (MalVec is offline)

### Recommended Deployment

For production deployment analyzing potentially malicious samples:

1. **Run in VM**: Isolate MalVec in a disposable virtual machine
2. **Network isolation**: Disable network access to the VM
3. **Snapshot before analysis**: Restore VM state after each batch
4. **Monitor resources**: Alert on unusual CPU/memory usage
5. **Regular updates**: Keep LIEF and dependencies updated

## Security Testing

Run the security test suite:

```bash
# Run all security tests
pytest tests/security/ -v

# Run specific test categories
pytest tests/security/test_sandbox.py -v      # Sandbox tests
pytest tests/security/test_audit.py -v        # Audit logging tests
pytest tests/security/test_extractor_sandbox.py -v  # Extractor integration
```

### Test Coverage

The security tests verify:

- Timeout enforcement kills long-running processes
- Memory limits prevent exhaustion (on supported platforms)
- Exception propagation from sandboxed code
- Process crash detection
- File validation before processing
- Audit log structure and content
- Privacy features (hash-only logging)

## Audit Log Format

Audit logs are structured JSON for SIEM integration:

### Classification Event

```json
{
  "event": "classification",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "file": {
    "sha256": "abc123..."
  },
  "prediction": "MALWARE",
  "confidence": 0.95,
  "processing_time_ms": 1234,
  "needs_review": false
}
```

### Validation Failure

```json
{
  "event": "validation_failure",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "file": {
    "sha256": "def456..."
  },
  "reason": "File size exceeds limit",
  "error_type": "size_exceeded"
}
```

### Sandbox Violation

```json
{
  "event": "sandbox_violation",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "file": {
    "sha256": "ghi789..."
  },
  "violation": "Execution exceeded 30s timeout",
  "violation_type": "timeout"
}
```

## Incident Response

### If Sandbox Violation Detected

1. **Check audit logs**: Review `/var/log/malvec/audit.log` (or configured location)
2. **Identify file**: Use SHA256 hash from logs
3. **Isolate sample**: Move to quarantine for manual analysis
4. **Analyze in VM**: Use isolated environment for deeper inspection
5. **Report**: Document findings and update threat intelligence

### Log Locations

| Platform | Default Location |
|----------|-----------------|
| Linux | `/var/log/malvec/audit.log` |
| macOS | `/var/log/malvec/audit.log` |
| Windows | `%LOCALAPPDATA%\malvec\logs\audit.log` |

### Useful Commands

```bash
# View recent audit events
tail -f /var/log/malvec/audit.log

# Find all sandbox violations
grep "sandbox_violation" /var/log/malvec/audit.log

# Find all malware classifications
grep '"prediction": "MALWARE"' /var/log/malvec/audit.log | jq .

# Count classifications by type
grep "classification" /var/log/malvec/audit.log | \
  jq -r '.prediction' | sort | uniq -c
```

## Platform Considerations

### Linux

- Full memory limit support via `resource.setrlimit`
- Process isolation via `multiprocessing`
- Recommended for production deployments

### macOS

- Memory limits available but less effective
- Process isolation works correctly
- Suitable for development and testing

### Windows

- Memory limits not enforced (monitoring only)
- Process isolation works via 'spawn' mode
- Use VM isolation for production analysis

## Version History

| Version | Changes |
|---------|---------|
| 1.0 | Initial security hardening (Phase 8) |

## Contact

For security issues, please report via the project's issue tracker.
