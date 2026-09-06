@echo off
rem Ember local CLI wrapper.
rem Usage:  ember.cmd ingest <paths...>   |   ember.cmd ask "your question"   |   ember.cmd status
setlocal
rem mantle is imported as a package (`mantle.db.…`), so its OUTER src goes on the path, not
rem `src\mantle`. Mantle's modules are package-qualified, so the inner path half-loads and imports
rem the same class as two distinct objects.
rem Each entry below names a sibling checkout's outer src. An entry naming a directory that is not
rem there resolves anyway on a box whose venv supplies the package through an editable install, so
rem the path is worth checking against the tree rather than against a working run.
set "PYTHONPATH=%~dp0src;%~dp0..\agience-mantle\src;%~dp0..\entroptics\src"
rem The model/content cache gets large, so a box that wants it off the system drive names a
rem volume in EMBER_CACHE_DIR itself. Nothing here names a drive: the default sits under the
rem user profile, which exists on every Windows box, so the wrapper works on a fresh checkout.
if not defined EMBER_CACHE_DIR set "EMBER_CACHE_DIR=%USERPROFILE%\.cache\agience\ember"
set "HF_HUB_DISABLE_PROGRESS_BARS=1"
python -m ember.cli %*
endlocal
