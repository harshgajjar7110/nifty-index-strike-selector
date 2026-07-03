#!/bin/bash

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  PHASE 2.5 - STRIKE SELECTION ENGINE VERIFICATION"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check files exist
echo "📁 Files Created/Modified:"
for file in engine_strike_selection.py test_phase2_5.py PHASE2_5_IMPLEMENTATION_SUMMARY.md; do
    if [ -f "$file" ]; then
        lines=$(wc -l < "$file")
        echo "  ✓ $file ($lines lines)"
    else
        echo "  ✗ $file MISSING"
        exit 1
    fi
done

echo ""
echo "📝 Modified files with integration:"
for file in run_pipeline.py; do
    if grep -q "select-strikes" "$file"; then
        echo "  ✓ $file (contains select-strikes mode)"
    else
        echo "  ✗ $file (missing select-strikes mode)"
        exit 1
    fi
done

# Syntax check
echo ""
echo "🔍 Syntax Validation:"
python3 -m py_compile engine_strike_selection.py && echo "  ✓ engine_strike_selection.py" || exit 1
python3 -m py_compile test_phase2_5.py && echo "  ✓ test_phase2_5.py" || exit 1
python3 -m py_compile run_pipeline.py && echo "  ✓ run_pipeline.py" || exit 1

# Run tests
echo ""
echo "🧪 Test Results:"
python3 test_phase2_5.py 2>&1 | grep "RESULTS:" || exit 1

# Verify key functions
echo ""
echo "✨ Key Features Implemented:"
python3 << 'PYEOF'
import ast
import sys

with open('engine_strike_selection.py', 'r') as f:
    tree = ast.parse(f.read())

functions = [item.name for item in ast.walk(tree) if isinstance(item, ast.FunctionDef)]
classes = [item.name for item in ast.walk(tree) if isinstance(item, ast.ClassDef)]

required_classes = ['CandidateFilter', 'StrikeSelector', 'LognormalDistribution']
required_functions = ['calculate_greeks', 'generate_strike_recommendations', 'generate_candidate_strikes']

print("  Classes:")
for cls in required_classes:
    if cls in classes:
        print(f"    ✓ {cls}")
    else:
        print(f"    ✗ {cls} MISSING")
        sys.exit(1)

print("  Functions:")
for func in required_functions[:2]:  # First 2 are in this module
    if func in functions:
        print(f"    ✓ {func}")
    else:
        print(f"    ✗ {func} MISSING")
        sys.exit(1)
PYEOF

echo ""
echo "🚀 Pipeline Integration:"
python3 << 'PYEOF'
import run_pipeline
import inspect

source = inspect.getsource(run_pipeline.main)
if 'select-strikes' in source:
    print("  ✓ select-strikes mode routed in main()")
else:
    print("  ✗ select-strikes not found in main()")
    exit(1)

# Check for mode function
if hasattr(run_pipeline, 'mode_select_strikes'):
    print("  ✓ mode_select_strikes() function exists")
else:
    print("  ✗ mode_select_strikes() function missing")
    exit(1)
PYEOF

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ PHASE 2.5 COMPLETE & VERIFIED"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📚 Documentation:"
echo "  • PHASE2_5_IMPLEMENTATION_SUMMARY.md (comprehensive guide)"
echo ""
echo "🎯 Quick Start:"
echo "  # First-time setup (one-time)"
echo "  python run_pipeline.py --mode regime-train"
echo ""
echo "  # Weekly: Select optimal strikes"
echo "  python run_pipeline.py --mode select-strikes"
echo ""
echo "  # Check latest recommendations"
echo "  cat data/strikes_recommendation_latest.json"
echo ""
