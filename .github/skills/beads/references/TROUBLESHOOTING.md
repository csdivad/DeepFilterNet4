# Troubleshooting Guide

Common issues and their solutions when using bd.

## Installation Issues

### bd: command not found

**Problem**: bd is not in your PATH.

**Solution**:
```bash
# Check if installed
which bd

# If not found, install
cargo install beads

# Or add cargo bin to PATH
export PATH="$HOME/.cargo/bin:$PATH"
```

### Permission denied

**Problem**: Can't write to .beads directory.

**Solution**:
```bash
# Check permissions
ls -la .beads/

# Fix permissions
chmod 755 .beads/
chmod 644 .beads/beads.db
```

## Database Issues

### No database found

**Problem**: bd can't find the database.

**Solution**:
```bash
# Check if .beads exists
ls -la .beads/

# Initialize if needed
mkdir -p .beads
bd create "Initial issue" -p 3 --json
```

### Database locked

**Problem**: Another process has the database locked.

**Solution**:
```bash
# Check whether the Dolt server is already running
bd dolt status

# Restart the local Dolt server if needed
bd dolt stop
bd dolt start

# Or kill orphaned servers and retry
bd dolt killall
```

### Staleness Error

**Problem**: Database out of sync with JSONL.

**Solution**:
```bash
# Force import from JSONL
bd import -i .beads/issues.jsonl

# If import shows "0 created, 0 updated" but staleness persists
bd import --force -i .beads/issues.jsonl

# Emergency: skip staleness check
bd --allow-stale ready --json
```

### Database Corruption

**Problem**: SQLite database is corrupted.

**Solution**:
```bash
# Rebuild from JSONL (source of truth)
rm .beads/beads.db
bd import -i .beads/issues.jsonl

# If JSONL is also bad, recover from git
git checkout HEAD -- .beads/issues.jsonl
bd import -i .beads/issues.jsonl
```

## Sync Issues

### Dolt push/pull fails

**Problem**: `bd dolt pull` or `bd dolt push` fails.

**Solution**:
```bash
# Check whether a Dolt remote is configured
bd dolt remote list

# Check local Dolt configuration / connectivity
bd dolt show
bd dolt status

# If there are pending Dolt changes, commit them first
bd dolt commit -m "Sync beads state"

# Retry replication
bd dolt pull
bd dolt push
```

If `bd dolt remote list` shows no remotes, local Beads issue tracking still
works; only remote Dolt replication is unavailable until you add a remote.

### JSONL conflicts

**Problem**: Git merge conflict in issues.jsonl.

**Solution**:
```bash
# Accept both versions (JSONL is append-only friendly)
git checkout --ours .beads/issues.jsonl
git checkout --theirs .beads/issues.jsonl

# Or manually merge
# Each line is independent, take all unique lines

# Then import
bd import -i .beads/issues.jsonl
```

### Changes not replicated remotely

**Problem**: Changes made in Beads but not appearing on a Dolt remote.

**Solution**:
```bash
# Inspect local Dolt state
bd dolt status
bd dolt remote list

# If a remote exists, commit then push
bd dolt commit -m "Sync beads state"
bd dolt push

# If no remote exists, local Beads issue tracking is still working; add a
# Dolt remote before expecting remote replication.
```

## Dolt Server Issues

### Dolt server not starting

**Problem**: the local Beads Dolt server fails to start.

**Solution**:
```bash
# Check logs
bd dolt status

# Check for port conflicts
lsof -i :13357  # Replace with the port shown by `bd dolt show` if needed

# Start manually
bd dolt start
```

### Embedded mode concurrency warning

**Problem**: `bd doctor` warns about embedded-mode concurrency or a Dolt lock.

**Solution**:
```bash
# Confirm the Dolt server is running
bd dolt status

# Restart the server if needed
bd dolt stop
bd dolt start
```

### Sandbox mode not detected

**Problem**: bd not auto-detecting sandboxed environment.

**Solution**:
```bash
# Explicitly enable sandbox mode
bd --sandbox ready --json

# Or set environment
export BEADS_DOLT_AUTO_COMMIT=off
```

## Command Issues

### Invalid issue ID

**Problem**: Issue ID not recognized.

**Solution**:
```bash
# Check exact ID
bd list --json | jq '.[].id'

# IDs are case-sensitive
bd show bd-ABC123 --json  # Not bd-abc123
```

### Missing required fields

**Problem**: Create fails with missing field error.

**Solution**:
```bash
# Always quote titles
bd create "My issue title" -p 1 --json

# Include required flags
bd create "Title" -t bug -p 1 --json
```

### Dependency cycle

**Problem**: Can't add dependency due to cycle.

**Solution**:
```bash
# Check existing dependencies
bd dep tree <id>

# Remove conflicting dependency first
bd dep remove <from> <to>

# Then add new one
bd dep add <from> <to>
```

## Performance Issues

### Slow queries

**Problem**: bd commands taking too long.

**Solution**:
```bash
# Check database size
ls -lh .beads/beads.db

# Clean up closed issues
bd admin cleanup --older-than 90 --force --json

# Compact old issues
bd admin compact --auto --all --tier 1
```

### Memory issues

**Problem**: bd using too much memory.

**Solution**:
```bash
# Use pagination for large lists
bd list --limit 50 --json

# Process in batches
bd list --status open --json | head -100
```

## Migration Issues

### Old database format

**Problem**: Database from older bd version.

**Solution**:
```bash
# Run migration
bd migrate --dry-run  # Preview
bd migrate            # Execute

# If issues, check migration plan
bd migrate --inspect --json
```

### Missing fields after upgrade

**Problem**: New fields not present in old issues.

**Solution**:
```bash
# Export and reimport
bd export -o backup.jsonl
bd import -i backup.jsonl

# Or update individual issues
bd update <id> --design "" --json  # Initialize field
```

## Common Error Messages

### "Issue not found"

Issue ID doesn't exist in database.

```bash
# List all issues to find correct ID
bd list --json
```

### "Blocked by open issues"

Can't close issue that blocks open issues.

```bash
# Check what it blocks
bd dep tree <id>

# Either close blockers first or remove dependency
bd dep remove <id> <blocked-id>
```

### "Stale database"

Database differs from JSONL source of truth.

```bash
# Import to refresh
bd import -i .beads/issues.jsonl

# Or with force
bd import --force -i .beads/issues.jsonl
```

### "Dolt server connection refused"

Can't connect to the Beads Dolt server.

```bash
# Check if running
bd dolt status

# Restart if needed
bd dolt stop
bd dolt start

# Or inspect configuration
bd dolt show
```

### "Permission denied" on sync

Git push/pull failing.

```bash
# Check git remote
git remote -v

# Check credentials
git fetch origin

# Fix authentication
ssh -T git@github.com  # For SSH
git credential fill    # For HTTPS
```

## Debug Mode

For detailed debugging:

```bash
# Enable verbose logging
RUST_LOG=debug bd ready --json

# Check Dolt server log
tail -f .beads/dolt-server.log

# Get full system info
bd info --schema --json
```

## Getting Help

If issues persist:

1. Check bd version: `bd --version`
2. Check database info: `bd info --json`
3. Check Dolt server health: `bd dolt status`
4. Review Dolt log: `.beads/dolt-server.log`
5. Try sandbox mode: `bd --sandbox ready --json`

## See Also

- [REFERENCE.md](REFERENCE.md) - CLI commands, workflows, and patterns
- [ADVANCED.md](ADVANCED.md) - Power features and integration
