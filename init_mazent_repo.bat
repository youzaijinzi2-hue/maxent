@echo off
REM ==========================================================
REM Initialize MaxEnt repository with YAML + reproducibility
REM ==========================================================

echo Creating root-level files...
type nul > README.md
type nul > LICENSE
type nul > pyproject.toml
type nul > .gitignore
type nul > .gitattributes

REM ----------------------------------------------------------
REM configs (YAML: single source of truth)
REM ----------------------------------------------------------
echo Creating configs/ ...
mkdir configs
mkdir configs\dataset
mkdir configs\model
mkdir configs\run

type nul > configs\default.yaml
type nul > configs\dataset\xas_sample.yaml
type nul > configs\model\maxent.yaml
type nul > configs\run\debug.yaml

REM ----------------------------------------------------------
REM docs
REM ----------------------------------------------------------
echo Creating docs/ ...
mkdir docs
type nul > docs\theory.md
type nul > docs\design.md

REM ----------------------------------------------------------
REM examples
REM ----------------------------------------------------------
echo Creating examples/ ...
mkdir examples
type nul > examples\00_quickstart.ipynb
type nul > examples\01_xas_to_green_and_maxent.py

REM ----------------------------------------------------------
REM scripts (CLI entry points)
REM ----------------------------------------------------------
echo Creating scripts/ ...
mkdir scripts
type nul > scripts\run_maxent.py
type nul > scripts\prepare_from_xas.py

REM ----------------------------------------------------------
REM src/maxent (library code)
REM ----------------------------------------------------------
echo Creating src/maxent/ ...
mkdir src
mkdir src\maxent
type nul > src\maxent\__init__.py
type nul > src\maxent\config.py
type nul > src\maxent\types.py
type nul > src\maxent\plotting.py
type nul > src\maxent\reconstruct.py

REM ---- io ----
mkdir src\maxent\io
type nul > src\maxent\io\__init__.py
type nul > src\maxent\io\xas.py

REM ---- preprocess ----
mkdir src\maxent\preprocess
type nul > src\maxent\preprocess\__init__.py
type nul > src\maxent\preprocess\xas_preprocess.py

REM ---- kernels ----
mkdir src\maxent\kernels
type nul > src\maxent\kernels\__init__.py
type nul > src\maxent\kernels\fermion_tau.py
type nul > src\maxent\kernels\matsubara.py

REM ---- operators ----
mkdir src\maxent\operators
type nul > src\maxent\operators\__init__.py
type nul > src\maxent\operators\mapping.py
type nul > src\maxent\operators\forward.py

REM ---- mcmc ----
mkdir src\maxent\mcmc
type nul > src\maxent\mcmc\__init__.py
type nul > src\maxent\mcmc\sampler.py
type nul > src\maxent\mcmc\replica_exchange.py
type nul > src\maxent\mcmc\state.py

mkdir src\maxent\mcmc\moves
type nul > src\maxent\mcmc\moves\__init__.py
type nul > src\maxent\mcmc\moves\pca.py
type nul > src\maxent\mcmc\moves\weight_sharing.py
type nul > src\maxent\mcmc\moves\mala.py

REM ---- diagnostics ----
mkdir src\maxent\diagnostics
type nul > src\maxent\diagnostics\__init__.py
type nul > src\maxent\diagnostics\ess.py
type nul > src\maxent\diagnostics\rhat.py

REM ----------------------------------------------------------
REM tests
REM ----------------------------------------------------------
echo Creating tests/ ...
mkdir tests
type nul > tests\test_forward_consistency.py
type nul > tests\test_mapping_jacobian.py
type nul > tests\test_mcmc_invariants.py

REM ----------------------------------------------------------
REM data (raw data, not tracked)
REM ----------------------------------------------------------
echo Creating data/ ...
mkdir data
mkdir data\raw
mkdir data\processed

type nul > data\raw\.gitkeep
type nul > data\processed\.gitkeep

REM ----------------------------------------------------------
REM runs (experiment outputs, reproducibility bundle)
REM ----------------------------------------------------------
echo Creating runs/ ...
mkdir runs
type nul > runs\.gitkeep

REM ----------------------------------------------------------
REM Done
REM ----------------------------------------------------------
echo.
echo ==========================================================
echo MaxEnt reproducible repository structure created.
echo Next steps:
echo   1. git init
echo   2. git add .
echo   3. git commit -m "Initial reproducible project structure"
echo ==========================================================
