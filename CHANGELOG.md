# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

### [0.1.2](https://github.com/chime-experiment/dias/compare/0.1.0...0.1.2) (2019-09-20)


### Bug Fixes

* **find_jump_analyzer:** skip files that have a single time sample. ([b8d0367](https://github.com/chime-experiment/dias/commit/b8d0367))
* **flag_rfi_analyzer:** fixes typo log --> logger. ([#117](https://github.com/chime-experiment/dias/issues/117)) ([569e2ee](https://github.com/chime-experiment/dias/commit/569e2ee))
* **sensitivity:** export correct value in sensitivity metric ([45d98d3](https://github.com/chime-experiment/dias/commit/45d98d3))
* **setup.py:** Don't import dias in setup.py ([180573a](https://github.com/chime-experiment/dias/commit/180573a)), closes [#111](https://github.com/chime-experiment/dias/issues/111)


### Features

* **script:** print dias version string ([#120](https://github.com/chime-experiment/dias/issues/120)) ([0219d91](https://github.com/chime-experiment/dias/commit/0219d91)), closes [#106](https://github.com/chime-experiment/dias/issues/106)
* **sensitivity_analyzer**: change cadence to 2h


## Version 0.1.1 (2019-06-27)

### New Features
[SensitivityAnalyzer] Add sensitivity metric


## Version 0.1.0 (2019-06-13)

### New Features
- Sensitivity Analyzer
- [task] Add metric dias_task_no_data_total
- RFI Flagging Analyzer
- create dias.__version__ with versioneer
- Findjump Analyzer
- [CI] run scrips/dias configtest


### Bug Fixes
- [task] convert period to seconds in random start delay calculation
- Use pyyaml.safe_load to avoid a warning


### Documentation Changes
- [utils] replace non-ASCII character


## Version 0.0.2 (2019-04-05)
