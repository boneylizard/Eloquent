# Eloquent – Change Log

All notable changes to this project will be documented in this file.

This log is intentionally simple, human-readable, and focused on real user-facing progress.

---

## 2026-01-14

### Added

* Added **Variable Upscaling Options** (2x, 3x, 4x) to the chat interface and Image Gen tab.
* Added a **Model Selector** dropdown for choosing specific ESRGAN upscaler models (.pth).
* Added **"Visualize Scene"** functionality to prompt LLM to automatically generate images based on the current chat context.
* Added a **"Set BG"** button to set generated images as the chat background.
* Added an Upscaler Model Directory setting in **Settings > Local SD / EloDiffusion**.

### Fixed

* Fixed the **Chat Background Image** to cover the entire screen behind messages (previously obscured by the message scroll area).
* Fixed the **Upscaler logic** to correctly handle image paths and utilize the backend `UpscaleManager`.
* Resolved a crash (`useEffect` error) in the chat image component.
* Fixed API mode compatibility for the "Visualize Scene" feature.
* Fixed the **"Visualization failed"** error message to include a close (X) button, allowing it to be dismissed without a page reload.
* Improved the **"Back" button** behavior in Chat. It now removes only the last message (acting as an "Undo") rather than deleting the entire user/bot exchange.
* Added a **"Delete" (Trash/X)** button to **ALL** messages (User, Bot, System, and **Images**), giving you full control to remove any individual message from the history.
* **UI Polish**: Updated all message dismissal/delete buttons to be **neutral (white/gray)** by default to reduce visual clutter, only turning red on hover.

---

## 2026-01-13

### Added

* Added full support for **Chatterbox Turbo** as a text-to-speech engine.
* Chatterbox Turbo supports paralinguistic cues such as `[laugh]`, `[cough]`, and similar expressive markers.
* These cues are now passed through correctly and rendered in voice output.
* Added a manual **"Fix Parakeet Dependencies (Downgrade NumPy)"** button in Settings > Audio.
* Added a backend endpoint `POST /stt/fix-parakeet-numpy` to safely apply the Parakeet dependency fix.

### Fixed

* Patched Parakeet / NeMo installation and compatibility issues.
* The Parakeet installation process (`stt_service.py`) now automatically detects NeMo and forces a downgrade to `numpy<2` to prevent compatibility crashes.
* Improved handling of interrupted or partially completed Parakeet installs.
* Added explicit frontend alerts and backend warning logs instructing users to restart the backend after the fix is applied.
* Ensured the Parakeet engine loads correctly after dependency repair.

---

## 2026-01-12

### Added

* Added a KNOWN_ISSUES.md file to document known problems, limitations, and temporary workarounds.

### Fixed

* The **“Test Voice Settings”** button in the Audio tab now works.
* It plays a test voice using the currently selected text-to-speech engine (Kokoro or Chatterbox), allowing users to confirm their TTS configuration is working correctly.
* Fixed the Story Tracker so its contents are now correctly injected into the LLM context.
* The Story Tracker can be opened from the control panel, auto-filled by the AI, edited manually, or used in combination.
* Once closed, the tracked state (characters, inventory, locations, key events, etc.) is automatically formatted, persisted as structured continuity data, and injected into subsequent LLM messages.
* Fixed a bug where **Save & Regenerate** sometimes didn’t work after editing a message.
* **Save & Regenerate** now correctly sends the edited user prompt to the AI instead of the original text.
* Regeneration now always uses the edited text straight away.
* Fixed an issue where the AI could fail silently and return a blank response.
* API-based models are now more reliable and retry automatically if they briefly fail.

---

## [Unreleased]

### Added

*

### Changed

*

### Fixed

*
