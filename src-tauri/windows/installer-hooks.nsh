!include "nsDialogs.nsh"
!include "LogicLib.nsh"

Var MiridTtsEnabled
Var MiridSttEnabled
Var MiridTtsEngine
Var MiridSttEngine
Var MiridNanoGptApiKey
Var MiridTtsCheckbox
Var MiridSttCheckbox
Var MiridTtsEngineControl
Var MiridSttEngineControl
Var MiridNanoGptApiKeyControl

Page custom MiridAudioPageCreate MiridAudioPageLeave

Function MiridAudioPageCreate
  ${GetOptions} $CMDLINE "/UPDATE" $0
  ${IfNot} ${Errors}
    Abort
  ${EndIf}
  ${GetOptions} $CMDLINE "/P" $0
  ${IfNot} ${Errors}
    Abort
  ${EndIf}
  ${If} ${Silent}
    Abort
  ${EndIf}

  StrCpy $MiridTtsEnabled "1"
  StrCpy $MiridSttEnabled "1"
  StrCpy $MiridTtsEngine "kokoro"
  StrCpy $MiridSttEngine "whisper"
  StrCpy $MiridNanoGptApiKey ""

  !insertmacro MUI_HEADER_TEXT "Audio setup" "Choose the speech engines Mirid starts with."
  nsDialogs::Create 1018
  Pop $0
  ${If} $0 == error
    Abort
  ${EndIf}

  ${NSD_CreateLabel} 0 0 100% 22u "Audio is enabled by default. Local engines load when first used."
  Pop $0

  ${NSD_CreateCheckbox} 0 28u 100% 12u "Enable text-to-speech when Mirid opens"
  Pop $MiridTtsCheckbox
  ${NSD_Check} $MiridTtsCheckbox

  ${NSD_CreateLabel} 0 48u 42% 12u "Text-to-speech engine"
  Pop $0
  ${NSD_CreateDropList} 43% 45u 57% 100u ""
  Pop $MiridTtsEngineControl
  ${NSD_CB_AddString} $MiridTtsEngineControl "Kokoro (local)"
  ${NSD_CB_AddString} $MiridTtsEngineControl "Chatterbox (local)"
  ${NSD_CB_AddString} $MiridTtsEngineControl "Chatterbox Turbo (local)"
  ${NSD_CB_AddString} $MiridTtsEngineControl "Chatterbox Nano (local)"
  ${NSD_CB_AddString} $MiridTtsEngineControl "VoxCPM2 (local)"
  ${NSD_CB_AddString} $MiridTtsEngineControl "VoxCPM2 GGUF (local)"
  ${NSD_CB_AddString} $MiridTtsEngineControl "NanoGPT Qwen-3-TTS (API)"
  ${NSD_CB_SelectString} $MiridTtsEngineControl "Kokoro (local)"

  ${NSD_CreateCheckbox} 0 76u 100% 12u "Enable speech-to-text when Mirid opens"
  Pop $MiridSttCheckbox
  ${NSD_Check} $MiridSttCheckbox

  ${NSD_CreateLabel} 0 96u 42% 12u "Speech-to-text engine"
  Pop $0
  ${NSD_CreateDropList} 43% 93u 57% 100u ""
  Pop $MiridSttEngineControl
  ${NSD_CB_AddString} $MiridSttEngineControl "Whisper (local)"
  ${NSD_CB_AddString} $MiridSttEngineControl "Whisper 3 Turbo (local)"
  ${NSD_CB_AddString} $MiridSttEngineControl "Parakeet v2 English (local)"
  ${NSD_CB_AddString} $MiridSttEngineControl "Parakeet v3 Multilingual (local)"
  ${NSD_CB_AddString} $MiridSttEngineControl "Parakeet Chinese (local)"
  ${NSD_CB_AddString} $MiridSttEngineControl "Nemotron Speech (local)"
  ${NSD_CB_AddString} $MiridSttEngineControl "Moonshine Tiny (local)"
  ${NSD_CB_AddString} $MiridSttEngineControl "Parakeet.cpp GGUF (local)"
  ${NSD_CB_AddString} $MiridSttEngineControl "NanoGPT ASR (API)"
  ${NSD_CB_SelectString} $MiridSttEngineControl "Whisper (local)"

  ${NSD_CreateLabel} 0 124u 42% 12u "NanoGPT API key (optional)"
  Pop $0
  ${NSD_CreatePassword} 43% 121u 57% 13u ""
  Pop $MiridNanoGptApiKeyControl

  ${NSD_CreateLabel} 0 143u 100% 24u "Required only when a NanoGPT speech engine is selected. The key is handed to Mirid on first run, then this installer file is deleted."
  Pop $0

  nsDialogs::Show
FunctionEnd

Function MiridAudioPageLeave
  ${NSD_GetState} $MiridTtsCheckbox $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $MiridTtsEnabled "1"
  ${Else}
    StrCpy $MiridTtsEnabled "0"
  ${EndIf}

  ${NSD_GetState} $MiridSttCheckbox $0
  ${If} $0 == ${BST_CHECKED}
    StrCpy $MiridSttEnabled "1"
  ${Else}
    StrCpy $MiridSttEnabled "0"
  ${EndIf}

  ${NSD_GetText} $MiridTtsEngineControl $0
  StrCpy $MiridTtsEngine "kokoro"
  ${If} $0 == "Chatterbox (local)"
    StrCpy $MiridTtsEngine "chatterbox"
  ${ElseIf} $0 == "Chatterbox Turbo (local)"
    StrCpy $MiridTtsEngine "chatterbox_turbo"
  ${ElseIf} $0 == "Chatterbox Nano (local)"
    StrCpy $MiridTtsEngine "chatterbox_nano"
  ${ElseIf} $0 == "VoxCPM2 (local)"
    StrCpy $MiridTtsEngine "voxcpm"
  ${ElseIf} $0 == "VoxCPM2 GGUF (local)"
    StrCpy $MiridTtsEngine "voxcpm-gguf"
  ${ElseIf} $0 == "NanoGPT Qwen-3-TTS (API)"
    StrCpy $MiridTtsEngine "nanogpt-Qwen-3-TTS-1.7B"
  ${EndIf}

  ${NSD_GetText} $MiridSttEngineControl $0
  StrCpy $MiridSttEngine "whisper"
  ${If} $0 == "Whisper 3 Turbo (local)"
    StrCpy $MiridSttEngine "whisper3"
  ${ElseIf} $0 == "Parakeet v2 English (local)"
    StrCpy $MiridSttEngine "parakeet"
  ${ElseIf} $0 == "Parakeet v3 Multilingual (local)"
    StrCpy $MiridSttEngine "parakeet-v3"
  ${ElseIf} $0 == "Parakeet Chinese (local)"
    StrCpy $MiridSttEngine "parakeet-zh"
  ${ElseIf} $0 == "Nemotron Speech (local)"
    StrCpy $MiridSttEngine "nemotron"
  ${ElseIf} $0 == "Moonshine Tiny (local)"
    StrCpy $MiridSttEngine "moonshine"
  ${ElseIf} $0 == "Parakeet.cpp GGUF (local)"
    StrCpy $MiridSttEngine "parakeet-cpp"
  ${ElseIf} $0 == "NanoGPT ASR (API)"
    StrCpy $MiridSttEngine "nanogpt"
  ${EndIf}

  ${NSD_GetText} $MiridNanoGptApiKeyControl $MiridNanoGptApiKey
  ${If} $MiridNanoGptApiKey == ""
    ${If} $MiridTtsEnabled == "1"
    ${AndIf} $MiridTtsEngine == "nanogpt-Qwen-3-TTS-1.7B"
      MessageBox MB_ICONEXCLAMATION|MB_OK "Enter a NanoGPT API key or choose a local text-to-speech engine."
      Abort
    ${EndIf}
    ${If} $MiridSttEnabled == "1"
    ${AndIf} $MiridSttEngine == "nanogpt"
      MessageBox MB_ICONEXCLAMATION|MB_OK "Enter a NanoGPT API key or choose a local speech-to-text engine."
      Abort
    ${EndIf}
  ${EndIf}
FunctionEnd

!macro NSIS_HOOK_POSTINSTALL
  ${GetOptions} $CMDLINE "/UPDATE" $0
  ${If} ${Errors}
    ${If} $MiridTtsEnabled == ""
      StrCpy $MiridTtsEnabled "1"
    ${EndIf}
    ${If} $MiridSttEnabled == ""
      StrCpy $MiridSttEnabled "1"
    ${EndIf}
    ${If} $MiridTtsEngine == ""
      StrCpy $MiridTtsEngine "kokoro"
    ${EndIf}
    ${If} $MiridSttEngine == ""
      StrCpy $MiridSttEngine "whisper"
    ${EndIf}

    CreateDirectory "$LOCALAPPDATA\ai.mirid.desktop"
    Delete "$LOCALAPPDATA\ai.mirid.desktop\installer-audio.ini"
    WriteINIStr "$LOCALAPPDATA\ai.mirid.desktop\installer-audio.ini" "audio" "ttsEnabled" "$MiridTtsEnabled"
    WriteINIStr "$LOCALAPPDATA\ai.mirid.desktop\installer-audio.ini" "audio" "sttEnabled" "$MiridSttEnabled"
    WriteINIStr "$LOCALAPPDATA\ai.mirid.desktop\installer-audio.ini" "audio" "ttsEngine" "$MiridTtsEngine"
    WriteINIStr "$LOCALAPPDATA\ai.mirid.desktop\installer-audio.ini" "audio" "sttEngine" "$MiridSttEngine"
    WriteINIStr "$LOCALAPPDATA\ai.mirid.desktop\installer-audio.ini" "audio" "nanogptSttModel" "fun-asr-flash-2026-06-15"
    ${If} $MiridNanoGptApiKey != ""
      WriteINIStr "$LOCALAPPDATA\ai.mirid.desktop\installer-audio.ini" "audio" "nanoGptApiKey" "$MiridNanoGptApiKey"
    ${EndIf}
  ${EndIf}
!macroend
