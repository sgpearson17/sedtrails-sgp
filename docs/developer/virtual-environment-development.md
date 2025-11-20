# Development with Virtual Environment

## Issue: Schema Validation with Development Changes

When developing SedTRAILS and making changes to configuration schemas (like adding new tracer methods), you may encounter validation errors even though your schema changes are correct. This typically happens when the `sedtrails` command uses a globally installed version instead of your development version.

### Symptoms

- Schema validation fails for new features you've added to `*.schema.json` files
- Error messages like `"Additional properties are not allowed ('your_new_feature' was unexpected)"`
- Your changes work in tests but fail when running the actual `sedtrails` command
- The error traceback shows paths outside your development directory

### Root Cause

The issue occurs because:

1. **Multiple installations**: You may have both a globally installed SedTRAILS and a development version
2. **Wrong executable**: The `sedtrails` command in your PATH may point to the global installation
3. **Schema loading**: The validator loads schemas from the installed package, not your development files

### Solution

Always use the virtual environment's executable when testing development changes:

#### ✅ Correct Approach
```bash
# Use the full path to your virtual environment's executable
C:/your-project/.venv/Scripts/sedtrails.exe run -c "your-config.yaml"

# Or activate the virtual environment first
.venv/Scripts/Activate.ps1
sedtrails run -c "your-config.yaml"
```

#### ❌ Incorrect Approach
```bash
# This may use the globally installed version
sedtrails run -c "your-config.yaml"
```

### Development Workflow

1. **Make schema changes** in `src/sedtrails/config/*.schema.json`
2. **Reinstall in development mode** to update the package:
   ```bash
   pip install -e .
   ```
3. **Test using the virtual environment executable**:
   ```bash
   C:/your-project/.venv/Scripts/sedtrails.exe run -c "examples/your-test-config.yaml"
   ```

### Verification Steps

To verify you're using the correct installation:

```bash
# Check which sedtrails executable you're using
Get-Command sedtrails

# Check if your development package is properly installed
pip show sedtrails

# Verify the package location
python -c "import sedtrails; print(sedtrails.__file__)"
```

The output should show paths within your development directory, not in global Python packages.

### Quick Checklist for Plugin Development

- [ ] Made changes to schema files
- [ ] Reinstalled with `pip install -e .`
- [ ] Using virtual environment's executable
- [ ] Added required characteristics for new particle types
- [ ] Tested configuration validates without errors

### Common Gotchas

1. **Missing characteristics**: New particle types need corresponding characteristics definitions
2. **Schema references**: Ensure all `$ref` references in schemas point to existing definitions
3. **Virtual environment activation**: Always verify you're in the correct environment
4. **Case sensitivity**: Schema property names are case-sensitive

By following these guidelines, you'll avoid the common pitfall of testing against the wrong SedTRAILS installation and ensure your development changes are properly validated.

N.B. This section of the documentation was prepared with input from Copilot (Claude Sonnet 4).