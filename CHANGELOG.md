# Eloquent – Change Log

All notable changes to this project will be documented in this file.

This log is intentionally simple, human-readable, and focused on real user-facing progress.

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
