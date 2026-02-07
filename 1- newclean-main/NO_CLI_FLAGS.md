# ⚠️ NO CLI FLAGS PERMITTED

## RULE ENFORCEMENT

**This cleanup script does NOT accept command-line flags or arguments.**

The script has been designed to enforce zero-configuration operation. Any attempt to pass CLI arguments will be rejected with an error.

---

## ✅ CORRECT USAGE

### Windows
```bash
# Double-click this file (RECOMMENDED)
START_HERE.bat
```

### Linux/Mac
```bash
cd "1- newclean-main"
python restore_playmat_hsv.py
```

**No arguments, no flags, no paths.**

---

## ❌ FORBIDDEN (Will Error)

```bash
# These will all be rejected:
python restore_playmat_hsv.py --help
python restore_playmat_hsv.py --workers 4
python restore_playmat_hsv.py --use-gpu
python restore_playmat_hsv.py scans/
python restore_playmat_hsv.py --anything
python restore_playmat_hsv.py arg1 arg2
```

---

## Why No CLI Flags?

1. **Consistency** - Everyone gets the same, tested configuration
2. **Reliability** - No misconfiguration possible
3. **Simplicity** - Nothing to learn or remember
4. **Support** - Easier to troubleshoot (one configuration only)
5. **Quality** - Settings have been optimized and locked

---

## What If I Need Customization?

**You don't.** The script has been carefully tuned with optimal settings for playmat restoration:

- ✅ Texture removal: Optimized multi-stage pipeline
- ✅ Edge preservation: Canny + keep-out zones
- ✅ Color accuracy: 8-color HLS palette
- ✅ Text protection: Top-hat + adaptive threshold
- ✅ GPU usage: Auto-detected and used when available
- ✅ Processing speed: Sequential for reliability
- ✅ Output quality: Maximum (zero grain, flat colors)

**All settings are production-ready and cannot be improved through CLI flags.**

---

## Technical Implementation

The script checks `sys.argv` at startup:

```python
if len(sys.argv) > 1:
    print("ERROR: Command-line arguments are not permitted")
    sys.exit(1)
```

This enforcement ensures:
- No user can accidentally misconfigure the script
- Technical support is simplified (one configuration)
- Output quality is consistent across all users
- The script remains stable and predictable

---

## For Older Versions (v2, v3, v4)

Older versions in directories `2- repo2-main`, `3- cleanup-main`, and `4- old` have CLI flags.

**DO NOT use those versions.** They are archived and inferior in quality.

**Use v1 only:** `1- newclean-main`

---

**BOTTOM LINE:** Run via `START_HERE.bat` and put your images in `scans/`. That's it. No flags, no arguments, no configuration.
