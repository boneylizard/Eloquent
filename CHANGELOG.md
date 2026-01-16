# Eloquent – Change Log

All notable changes to this project will be documented in this file.

This log is intentionally simple, human-readable, and focused on real user-facing progress.

---

## 2026-01-16

### Added

* **Conversation Summary & Save Point**: Implemented a "Summarize" feature in the chat header. Users can now generate a summary of their current conversation, save it, and load it into a fresh chat.
* **Context Injection**: Saved summaries are intelligently injected into the Prompt System, allowing the AI to "remember" past events even in new sessions or after a context reset.

* **Devstral Large (OpenRouter) Integration**: Fully integrated support for running Devstral 2 123b Large via OpenRouter API within the Code Editor.
* **Smart Model Routing**: The prompt engine now intelligently switches between local models (e.g., `devstral-small`) and external API models based on user selection.
* **Session Controls**: Added a "New Chat" button and improved "Delete" buttons (styled as a persistent white 'X') to the Code Editor sidebar for better session management.
* **Native Tool Execution**: Implemented a robust, brace-counting JSON parser (`_extract_balanced_json`) in `devstral_service.py`. This replaced the fragile regex parser, enabling "super fast" and reliable execution of file operations, even for complex code blocks with nested structures.

### Fixed

* **Summary Injection Bug**: Fixed a stale closure issue where loaded summaries were visible in the UI but not actually sent to the AI backend.
* **"Input Too Large" API Errors**: Fixed a critical bug in the context pruning logic. The system now correctly detects and truncates context even when the conversation history is a single massive block (regex failure scenario), preventing 400 Bad Request errors from upstream APIs.
* **Context Window Validity**: Verified and adjusted the backend logic to ensure the `Context Window` setting (e.g., 8192) is strictly respected, with a configurable safety margin (now 1000 tokens) to ensure stable generation.

* **Backend Crash Loop**: Fixed a critical `IndentationError` in `main.py` that was preventing the backend from started.
* **Local Model Priority**: Resolved a routing bug where local model requests were incorrectly falling back to the legacy external API endpoint. Local models now correctly take precedence when loaded.
* **"No Model" UI Bug**: Fixed an issue where the Code Editor incorrectly displayed "No Model" when an API model was selected. It now correctly identifies API models as "Ready".
* **Prompt Hardening**: Updated the `DEVSTRAL_SYSTEM_PROMPT` to strictly forbid the model from "simulating" edits. This forces the model to use the `write_file` tool, eliminating hallucinations and ensuring changes are actually saved to disk.

---

## 2026-01-15

### Added

* Added full support for `{{char}}` and `{{user}}` tag substitution in all character-related text fields.
* Added support for tag substitution in **User Profile** (Direct Injection), allowing the AI to know your name even when it's just a variable in the profile.
* Added support for tag substitution in **World Knowledge (Lore)** entries.
* Added **Director Mode** to the Choice Generator, allowing users to toggle between "Character Actions" and "Narrative Beats" (plot steering).
* Added **Emoji Support** to all AI-generated choices for better visual scanning.
* Added **OOC (Director Note) Injection**: Narrative beats are now injected as `(Director: ...)` notes to steer the AI without forcing a character action.
- **Choice Generator Reliability**: Hardened JSON parsing and instructions to prevent "Unexpected token" errors and ensure valid AI output.
- **Fixed Story Tracker Crash**: Resolved a reference error that caused the Story Tracker to crash the app after recent updates.
- **Standardized Story Tracker Headers**: Lore and Story context now use high-priority instructional headers (e.g., `[STORY TRACKER - Essential continuity guidance for this response]`) to improve AI adherence to world rules and plot state.
- **Story Tracker Save Button**: Added an explicit "Save Changes" button to the Story Tracker for clear visual confirmation of state persistence.
* Added a **Scene Summary** field to the Story Tracker. This persistent context grounds the AI in the current scene and mood.
* Added **AI Auto-Detection for Scene Summaries**: The Story Tracker analysis now automatically summarizes the current situation from chat history.
* Added **Instructional World Knowledge**: Enhanced lore injection to use a more instructional header `[WORLD KNOWLEDGE - Essential lore guidance for this response]`, ensuring the AI treats lore entries as direct guidance similar to the Author's Note.

### Fixed

* Fixed **Regenerate Response** to correctly use the updated character data, ensuring tags like `{{user}}` are properly replaced during regeneration.
* Fixed **Regenerate Variant** (swiping) to correctly include all context (Lore, User Profile) and perform tag substitutions, resolving an issue where swipes would lose this critical information.
* Refactored the prompt generation logic to use a centralized system, preventing future inconsistencies between different generation modes.
* Deeply integrated **Story Tracker** context into the main generation stream and swipe (variant) logic, ensuring total continuity.

---

## 2026-01-14

### Added

* Added **Variable Upscaling Options** (2x, 3x, 4x) to the chat interface and Image Gen tab.
* Added a **Model Selector** dropdown for choosing specific ESRGAN upscaler models (.pth).
* Added **"Visualize Scene"** functionality to prompt LLM to automatically generate images based on the current chat context.
* Added a **"Set BG"** button to set generated images as the chat background.
* Added an Upscaler Model Directory setting in **Settings > Local SD / EloDiffusion**.
* Added **Paralinguistic Tag Normalization** for Chatterbox Turbo, enabling support for expressive tags like `[laugh]`, `[sigh]`, and `[clear throat]`.

### Fixed

* Fixed the **Chat Background Image** to cover the entire screen behind messages (previously obscured by the message scroll area).
* Fixed the **Upscaler logic** to correctly handle image paths and utilize the backend `UpscaleManager`.
* Resolved a crash (`useEffect` error) in the chat image component.
* Fixed API mode compatibility for the "Visualize Scene" feature.
* Fixed the **"Visualization failed"** error message to include a close (X) button, allowing it to be dismissed without a page reload.
* Improved the **"Back" button** behavior in Chat. It now removes only the last message (acting as an "Undo") rather than deleting the entire user/bot exchange.
* Added a **"Delete" (Trash/X)** button to **ALL** messages (User, Bot, System, and **Images**), giving you full control to remove any individual message from the history.
* **UI Polish**: Updated all message dismissal/delete buttons to be **neutral (white/gray)** by default to reduce visual clutter, only turning red on hover.
* Fixed significant **Typing Lag** in the Chat interface.
* Optimized the Chat component by memoizing message rendering (`ChatMessage`) and stabilizing event handlers. Inputting text no longer triggers re-rendering of the entire message history, keeping the UI snappy even in long conversations.
* Fixed **Invisible Chat Buttons** where action buttons (Regenerate, Delete) were hidden on hover.
* Fixed **Disappearing AI Responses**: Resolved a critical bug where responses starting with "Assistant:" were being deleted by the text cleaner.
* Fixed- **OpenAI API Context Errors:** Added intelligent context pruning (preserves system + last user message) to prevent "input too long" errors. Default limit is 8192 tokens, but this is now **configurable per custom endpoint** in Settings > LLM Settings.
* Fixed **Chatterbox Turbo Voice Cloning**: Restored correct voice reference passing for Chatterbox engines.
* Fixed **Streaming TTS Tag Stripping**: Corrected regex logic to preserve paralinguistic tags (e.g., `[laugh]`, `[clear_throat]`) during streaming playback.

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
