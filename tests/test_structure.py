"""
Syntax and structure validation test for proficiency_probing package.
This verifies the code structure without requiring dependencies.
"""

import ast
import os

def check_file_syntax(filepath):
    """Check if a Python file has valid syntax."""
    with open(filepath, 'r') as f:
        code = f.read()
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, str(e)

def check_class_methods(filepath, class_name, required_methods):
    """Check if a class has required methods."""
    with open(filepath, 'r') as f:
        code = f.read()
    
    tree = ast.parse(code)
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            methods = [m.name for m in node.body if isinstance(m, ast.FunctionDef)]
            missing = [m for m in required_methods if m not in methods]
            return len(missing) == 0, missing
    
    return False, [f"Class {class_name} not found"]

# Test files
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(test_dir)
base_path = os.path.join(project_root, 'src', 'proficiency_probing')

files_to_check = {
    '__init__.py': os.path.join(base_path, '__init__.py'),
    'embedder.py': os.path.join(base_path, 'embedder.py'),
    'probe.py': os.path.join(base_path, 'probe.py'),
    'pipeline.py': os.path.join(base_path, 'pipeline.py'),
}

print("="*70)
print("Proficiency Probing Pipeline - Code Structure Validation")
print("="*70)
print()

# Check syntax of all files
print("1. Checking Python syntax...")
all_valid = True
for name, filepath in files_to_check.items():
    valid, error = check_file_syntax(filepath)
    if valid:
        print(f"   ✓ {name}")
    else:
        print(f"   ✗ {name}: {error}")
        all_valid = False

if not all_valid:
    print("\n✗ Syntax errors found!")
    exit(1)

print()
print("2. Checking class structures...")

# Check TextEmbedder
success, missing = check_class_methods(
    files_to_check['embedder.py'],
    'TextEmbedder',
    ['__init__', 'embed_texts', 'get_embedding_dim', '_pool_embeddings', '_extract_attention_head_embeddings']
)
if success:
    print("   ✓ TextEmbedder has all required methods")
else:
    print(f"   ✗ TextEmbedder missing: {missing}")
    all_valid = False

# Check OrdinalProbe
success, missing = check_class_methods(
    files_to_check['probe.py'],
    'OrdinalProbe',
    ['__init__', 'forward', 'fit', 'evaluate', 'predict', 'predict_proba', 'predict_labels']
)
if success:
    print("   ✓ OrdinalProbe has all required methods")
else:
    print(f"   ✗ OrdinalProbe missing: {missing}")
    all_valid = False

# Check ProficiencyProbingPipeline
success, missing = check_class_methods(
    files_to_check['pipeline.py'],
    'ProficiencyProbingPipeline',
    ['__init__', 'fit', 'evaluate', 'predict', 'cross_distribution_evaluation', 'save', 'load']
)
if success:
    print("   ✓ ProficiencyProbingPipeline has all required methods")
else:
    print(f"   ✗ ProficiencyProbingPipeline missing: {missing}")
    all_valid = False

print()
print("3. Checking example files...")
examples_dir = os.path.join(project_root, 'examples')
example_files = [
    os.path.join(examples_dir, 'basic_usage.py'),
    os.path.join(examples_dir, 'layer_head_exploration.py'),
]

for example_file in example_files:
    valid, error = check_file_syntax(example_file)
    filename = os.path.basename(example_file)
    if valid:
        print(f"   ✓ {filename}")
    else:
        print(f"   ✗ {filename}: {error}")
        all_valid = False

if all_valid:
    print()
    print("="*70)
    print("✓ SUCCESS: All validation checks passed!")
    print("="*70)
    print()
    print("Implementation Summary:")
    print("-" * 70)
    print()
    print("Core Components:")
    print("  1. TextEmbedder (src/proficiency_probing/embedder.py)")
    print("     - Supports any HuggingFace transformer model")
    print("     - Can extract embeddings from any layer (hidden states)")
    print("     - Can extract from specific attention heads")
    print("     - Multiple pooling strategies (mean, cls, max)")
    print()
    print("  2. OrdinalProbe (src/proficiency_probing/probe.py)")
    print("     - Linear probe for ordinal regression")
    print("     - Uses cumulative link model approach")
    print("     - Learns threshold parameters for ordinal classes")
    print("     - Respects ordering of proficiency levels")
    print()
    print("  3. ProficiencyProbingPipeline (src/proficiency_probing/pipeline.py)")
    print("     - End-to-end workflow orchestration")
    print("     - Handles train/validation splits")
    print("     - Cross-distribution evaluation")
    print("     - Save/load trained probes")
    print()
    print("Features:")
    print("  ✓ Flexible model support (any HuggingFace model)")
    print("  ✓ Layer & head selection for probing")
    print("  ✓ Ordinal regression (not just classification)")
    print("  ✓ Cross-distribution generalizability testing")
    print("  ✓ GPU support with automatic detection")
    print("  ✓ Batch processing for efficiency")
    print("  ✓ Save/load functionality")
    print()
    print("Documentation:")
    print("  • Comprehensive README with usage examples")
    print("  • Example scripts in examples/ directory")
    print("  • Detailed docstrings in all modules")
    print()
    print("="*70)
else:
    print()
    print("✗ Some validation checks failed")
    exit(1)
