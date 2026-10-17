# Provenance and publication scope

AIE Optimizer grew out of research alongside
[AIE-PerfModel](https://github.com/lxy-littlechai/AIE-PerfModel).
This repository preserves the Optimizer package name and the sibling
`Model/Timing` layout used by its Analyzer bridge.

The bundled `Model/Timing` files and `Model/analysis_errors.py` are the local
Analyzer support extensions required by this Optimizer snapshot. They include
clocked causal constraints, buffer lifecycle, component cost coverage, and
compiled resource evidence parsing. They are a dependency subset, not a complete
copy of the original Analyzer or a promise of AIE-ML performance coverage.

The public tree contains source code, templates, documentation, and synthetic
test fixtures. It excludes private ORCA sources, vendor SDK headers and
libraries, platform packages, credentials, server configuration, raw target AST
captures, and private experiment archives.

No repository-wide open-source license is assigned in this initial publication.
No upstream or third-party license is replaced or expanded by this notice.
Python packages, the compiler, and AMD tools must be obtained separately under
their respective terms.
