# ⚠️ ARCHIVED VERSION

This is an **archived version** of the cleanup script.

## Please use the latest version instead:

👉 **[../1- newclean-main/](../1-%20newclean-main/)**

---

## Why upgrade to v1?

The latest version (1-newclean-main) has:

✅ **Better texture removal** - Multi-stage pipeline vs. single bilateral filter  
✅ **Cleaner edges** - Canny keep-out zones prevent bleeding  
✅ **More accurate colors** - HLS color space handles lighting better  
✅ **Simpler to use** - Zero configuration, just run it  
✅ **More reliable** - No memory thrashing from parallel workers  
✅ **Better output quality** - 8 exact colors, zero grain  

---

## Migration

1. Copy your scans to: `../1- newclean-main/scans/`
2. Run: `cd "../1- newclean-main" && python restore_playmat_hsv.py`
3. Get output from: `../1- newclean-main/scans/output/`

---

See [../CLEANUP_SCRIPT_PROGRESSION.md](../CLEANUP_SCRIPT_PROGRESSION.md) for detailed version comparison.
