# Mirid release testing

## Manual clean-install test

Use a disposable Windows account. From that account, preview the exact state that will be removed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\reset_mirid_clean_install.ps1 -ExpectedUser tv
```

Then perform the reset deliberately:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\reset_mirid_clean_install.ps1 -ExpectedUser tv -ConfirmReset
```

Run the candidate installer only after the reset finishes. This removes the installed app, current and legacy app data, and backend settings, so the next launch must download and extract the full runtime again. Do not use this command in your primary Windows account.

## GitHub clean Windows VM

Run the `Mirid clean Windows install` workflow manually in `boneylizard/Eloquent`. Supply a direct candidate installer URL, its SHA-256 hash, and the expected version. The workflow uses a new `windows-2025` VM, refuses pre-existing Mirid state, performs a silent per-user install, launches without a GPU, downloads and extracts the runtime without a cache, verifies the exact PyInstaller layout, checks backend, TTS and WebView health, and uploads only logs plus a small JSON summary.

Run the quick checks before rebuilding the desktop installer:

```powershell
cd frontend
npm test
npm run test:ui-smoke
npm run release:check
cd ..\src-tauri
cargo test
```

After the release executable is built, replay a genuine first launch without uninstalling Mirid or losing your normal profile:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\test_first_run_release.ps1 -Mode Full -Interactive
```

The script temporarily moves Mirid's app data, legacy migration state, and backend settings into timestamped QA backups. It launches Mirid with a clean profile, waits for the runtime download and services, checks the first-run provider screen and dark theme, then lets you test manually. Press Enter in PowerShell when finished; the disposable profile is removed and your normal data is restored.

Use `-Mode RuntimeOnly` to repeat the runtime download and extraction while keeping the current WebView profile and onboarding state. Omit `-Interactive` for an unattended smoke test.

Unattended release scripts set the process-local `MIRID_QA_AUTO_BEGIN_SETUP=1` escape hatch so the runtime test can proceed without clicking through the purpose screen. Normal installs never set this variable and remain paused until the user makes a choice.

Each run writes a screenshot, DOM snapshot, logs, executable hash, result, and duration under `artifacts\first-run\`.

After every successful installer build, place the user-facing copies on the Desktop:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\export_installers.ps1
```

The newest installer is always `Desktop\Mirid Installers\Latest\Mirid-Setup.exe`. Timestamped copies remain under `Desktop\Mirid Installers\Releases\`.

## Genuine fresh Windows account

For the final release gate, double-click `Desktop\Mirid Installers\PREPARE FRESH WINDOWS TEST.cmd`. Alternatively, run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\prepare_fresh_windows_test.ps1
```

Approve the single Windows administrator prompt. The script creates `MiridReleaseTest`, copies the installer to `C:\Users\Public\Documents\Mirid Release Test\Mirid-Setup.exe`, and prints the test password. It does not install or launch Mirid.

Sign out, enter the fresh account, and test the complete download, extraction, onboarding, first launch, shutdown, restart, and uninstall journey. Do not copy any Mirid settings or app data into the account before the test.

After closing Mirid, run `C:\Users\Public\Documents\Mirid Release Test\COLLECT FRESH TEST RESULT.cmd`. It records the installed version and hash, runtime marker, runtime directory, latest launch log, service-ready state, and any recurrence of the access-denied setup failure in `fresh-test-result.json`. The collector does not launch or stop Mirid.

## Publish the verified installer

Stage and inspect the exact public release without uploading it:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\publish_installer_release.ps1
```

Only after the fresh-account test passes, publish it:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\publish_installer_release.ps1 -Publish -AllowUnsigned
```

The publisher uploads the versioned installer and checksums, then downloads the public file again and verifies its size and SHA-256 hash. It does not modify or deploy the Mirid website. Omit `-AllowUnsigned` once Mirid has a valid Windows code-signing certificate.
