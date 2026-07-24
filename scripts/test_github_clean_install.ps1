[CmdletBinding(DefaultParameterSetName = "RemoteInstaller")]
param(
    [Parameter(Mandatory = $true, ParameterSetName = "RemoteInstaller")]
    [ValidateNotNullOrEmpty()]
    [string]$InstallerUrl,
    [Parameter(Mandatory = $true, ParameterSetName = "LocalInstaller")]
    [ValidateNotNullOrEmpty()]
    [string]$InstallerPath,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[a-fA-F0-9]{64}$")]
    [string]$InstallerSha256,
    [string]$ExpectedVersion = "1.0.12",
    [string]$ExpectedRuntimeVersion = "v9",
    [int]$TimeoutSeconds = 14400,
    [string]$EvidenceDirectory = (Join-Path $PSScriptRoot "..\artifacts\github-clean-install")
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$evidence = [System.IO.Path]::GetFullPath($EvidenceDirectory)
$installer = Join-Path $env:RUNNER_TEMP "Mirid-Setup.exe"
$installedExecutable = Join-Path $env:LOCALAPPDATA "Mirid\mirid.exe"
$appDataRoot = Join-Path $env:LOCALAPPDATA "ai.mirid.desktop"
$installerAudioConfig = Join-Path $appDataRoot "installer-audio.ini"
$runtimeRoot = Join-Path $appDataRoot "runtime"
$runtimeMarker = Join-Path $runtimeRoot "runtime.ready"
$logRoot = Join-Path $appDataRoot "logs"
$devToolsPort = 9229
$startedAt = Get-Date
$downloadSeconds = 0
$firstLaunchSeconds = 0
$failure = ""
$passed = $false
$process = $null
$servicePorts = $null
$frontendServiceEndpoints = $null
$serviceListenerProcessIds = @()
$gracefulShutdownVerified = $false
$defaultTtsPortReservation = $null
$previousQaAutoBeginSetup = $env:MIRID_QA_AUTO_BEGIN_SETUP
$previousBrowserArguments = $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS

function Stop-MiridProcesses {
    Get-Process -Name "mirid", "mirid-sidecar-x86_64-pc-windows-msvc", "eloquent-sidecar-x86_64-pc-windows-msvc" -ErrorAction SilentlyContinue |
        Stop-Process -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
}

function Get-FreeDiskGiB {
    $drive = Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='C:'"
    return [math]::Round($drive.FreeSpace / 1GB, 2)
}

function Get-MiridServicePorts {
    if (-not (Test-Path -LiteralPath $logRoot)) {
        return $null
    }
    $latestLog = Get-ChildItem -LiteralPath $logRoot -Filter "*.log" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if (-not $latestLog) {
        return $null
    }
    $text = Get-Content -LiteralPath $latestLog.FullName -Raw
    $backendMatches = [regex]::Matches($text, "Starting backend sidecar on \S+:(\d+)")
    $ttsMatches = [regex]::Matches($text, "Starting tts sidecar on \S+:(\d+)")
    if (-not $backendMatches.Count -or -not $ttsMatches.Count) {
        return $null
    }
    return [pscustomobject]@{
        Backend = [int]$backendMatches[$backendMatches.Count - 1].Groups[1].Value
        Tts = [int]$ttsMatches[$ttsMatches.Count - 1].Groups[1].Value
    }
}

function Reserve-DefaultTtsPort {
    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Loopback,
            8002
        )
        $listener.Server.ExclusiveAddressUse = $true
        $listener.Start()
        return $listener
    } catch {
        if ($listener) {
            $listener.Stop()
        }
        throw
    }
}

function Get-MiridFrontendTarget {
    try {
        $targets = @(Invoke-RestMethod "http://127.0.0.1:$devToolsPort/json" -TimeoutSec 5)
        return $targets | Where-Object {
            $_.type -eq "page" -and $_.url -like "http://tauri.localhost/*"
        } | Select-Object -First 1
    } catch {
        return $null
    }
}

function Invoke-ChromeDevToolsExpression {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WebSocketDebuggerUrl,
        [Parameter(Mandatory = $true)]
        [string]$Expression,
        [int]$DeadlineSeconds = 10
    )

    $socket = [System.Net.WebSockets.ClientWebSocket]::new()
    $cancellation = [System.Threading.CancellationTokenSource]::new()
    $cancellation.CancelAfter([TimeSpan]::FromSeconds($DeadlineSeconds))
    try {
        $socket.ConnectAsync(
            [Uri]$WebSocketDebuggerUrl,
            $cancellation.Token
        ).GetAwaiter().GetResult()

        $messageId = 1
        $request = [ordered]@{
            id = $messageId
            method = "Runtime.evaluate"
            params = [ordered]@{
                expression = $Expression
                returnByValue = $true
                awaitPromise = $true
            }
        } | ConvertTo-Json -Depth 5 -Compress
        $requestBytes = [System.Text.Encoding]::UTF8.GetBytes($request)
        $socket.SendAsync(
            [System.ArraySegment[byte]]::new($requestBytes),
            [System.Net.WebSockets.WebSocketMessageType]::Text,
            $true,
            $cancellation.Token
        ).GetAwaiter().GetResult()

        $buffer = New-Object byte[] 65536
        while ($true) {
            $message = [System.IO.MemoryStream]::new()
            try {
                do {
                    $received = $socket.ReceiveAsync(
                        [System.ArraySegment[byte]]::new($buffer),
                        $cancellation.Token
                    ).GetAwaiter().GetResult()
                    if ($received.MessageType -eq [System.Net.WebSockets.WebSocketMessageType]::Close) {
                        throw "Chrome DevTools closed the connection before returning the evaluation result."
                    }
                    $message.Write($buffer, 0, $received.Count)
                } while (-not $received.EndOfMessage)

                $responseText = [System.Text.Encoding]::UTF8.GetString($message.ToArray())
            } finally {
                $message.Dispose()
            }

            $response = $responseText | ConvertFrom-Json
            if ($response.PSObject.Properties.Name -notcontains "id" -or $response.id -ne $messageId) {
                continue
            }
            if ($response.PSObject.Properties.Name -contains "error") {
                throw "Chrome DevTools evaluation failed: $($response.error.message)"
            }
            if ($response.result.PSObject.Properties.Name -contains "exceptionDetails") {
                throw "The frontend endpoint probe raised an exception."
            }
            return $response.result.result.value
        }
    } finally {
        $socket.Dispose()
        $cancellation.Dispose()
    }
}

function Wait-ForFrontendServiceEndpoints {
    param(
        [Parameter(Mandatory = $true)]
        [pscustomobject]$ExpectedPorts,
        [int]$DeadlineSeconds = 60
    )

    $deadline = (Get-Date).AddSeconds($DeadlineSeconds)
    $lastError = ""
    while ((Get-Date) -lt $deadline) {
        $target = Get-MiridFrontendTarget
        if ($target -and $target.webSocketDebuggerUrl) {
            try {
                $rawConfig = Invoke-ChromeDevToolsExpression `
                    -WebSocketDebuggerUrl $target.webSocketDebuggerUrl `
                    -Expression "JSON.stringify(window.__MIRID_SERVICE_ENDPOINTS__ || null)" `
                    -DeadlineSeconds 5
                $config = $rawConfig | ConvertFrom-Json
                if ($config) {
                    $expectedBackend = "http://127.0.0.1:$($ExpectedPorts.Backend)"
                    $expectedTts = "http://127.0.0.1:$($ExpectedPorts.Tts)"
                    if (
                        [int]$config.backendPort -eq $ExpectedPorts.Backend -and
                        [int]$config.ttsPort -eq $ExpectedPorts.Tts -and
                        $config.backend -eq $expectedBackend -and
                        $config.tts -eq $expectedTts
                    ) {
                        return $config
                    }
                    $lastError = "The frontend exposed backend $($config.backend) and TTS $($config.tts); expected $expectedBackend and $expectedTts."
                }
            } catch {
                $lastError = $_.Exception.Message
            }
        }
        Start-Sleep -Seconds 1
    }
    throw "The live frontend did not adopt Mirid's selected service endpoints. $lastError"
}

function Get-ServiceListenerProcessIds {
    param(
        [Parameter(Mandatory = $true)]
        [int[]]$Ports
    )

    $owners = @()
    foreach ($port in $Ports) {
        $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
        if (-not $listener) {
            throw "No listening process owns Mirid service port $port."
        }
        $owners += [int]$listener.OwningProcess
    }
    return @($owners | Sort-Object -Unique)
}

function Test-LoopbackPortBindable {
    param(
        [Parameter(Mandatory = $true)]
        [int]$Port
    )

    $listener = $null
    try {
        $listener = [System.Net.Sockets.TcpListener]::new(
            [System.Net.IPAddress]::Loopback,
            $Port
        )
        $listener.Server.ExclusiveAddressUse = $true
        $listener.Start()
        return $true
    } catch {
        return $false
    } finally {
        if ($listener) {
            $listener.Stop()
        }
    }
}

function Stop-MiridNormallyAndVerifyCleanup {
    param(
        [Parameter(Mandatory = $true)]
        [Diagnostics.Process]$DesktopProcess,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Ports,
        [Parameter(Mandatory = $true)]
        [int[]]$ListenerProcessIds,
        [int]$DeadlineSeconds = 60
    )

    $DesktopProcess.Refresh()
    if ($DesktopProcess.HasExited) {
        throw "Mirid exited before the normal-shutdown cleanup test."
    }
    if (-not $DesktopProcess.CloseMainWindow()) {
        throw "Mirid did not accept a normal main-window close request."
    }
    if (-not $DesktopProcess.WaitForExit($DeadlineSeconds * 1000)) {
        throw "Mirid did not exit after its main window was closed."
    }

    $deadline = (Get-Date).AddSeconds($DeadlineSeconds)
    while ((Get-Date) -lt $deadline) {
        $remainingOwners = @($ListenerProcessIds | Where-Object {
            Get-Process -Id $_ -ErrorAction SilentlyContinue
        })
        $backendReleased = Test-LoopbackPortBindable -Port $Ports.Backend
        $ttsReleased = Test-LoopbackPortBindable -Port $Ports.Tts
        if ($remainingOwners.Count -eq 0 -and $backendReleased -and $ttsReleased) {
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "Normal Mirid shutdown did not release its service processes and loopback listeners."
}

function Wait-ForMirid {
    param([Diagnostics.Process]$DesktopProcess, [int]$DeadlineSeconds)
    $deadline = (Get-Date).AddSeconds($DeadlineSeconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 10
        $DesktopProcess.Refresh()
        if ($DesktopProcess.HasExited) {
            throw "Mirid exited during first launch with code $($DesktopProcess.ExitCode)."
        }

        $runtimeReady = (Test-Path -LiteralPath $runtimeMarker) -and
            ((Get-Content -LiteralPath $runtimeMarker -Raw).Trim() -eq $ExpectedRuntimeVersion)
        if (-not $runtimeReady) { continue }

        $ports = Get-MiridServicePorts
        if (-not $ports) { continue }
        try {
            $backend = Invoke-RestMethod "http://127.0.0.1:$($ports.Backend)/health" -TimeoutSec 5
            $backendReady = $backend.status -eq "healthy"
        } catch {
            $backendReady = $false
        }
        try {
            $tts = Invoke-RestMethod "http://127.0.0.1:$($ports.Tts)/health" -TimeoutSec 5
            $ttsReady = $tts.status -eq "healthy"
        } catch {
            $ttsReady = $false
        }
        $frontendReady = [bool](Get-MiridFrontendTarget)

        if ($backendReady -and $ttsReady -and $frontendReady) {
            return $ports
        }
    }
    throw "Mirid did not finish first launch within $DeadlineSeconds seconds."
}

New-Item -ItemType Directory -Path $evidence -Force | Out-Null
Stop-MiridProcesses

try {
    foreach ($path in @(
        (Join-Path $env:LOCALAPPDATA "Mirid"),
        $appDataRoot,
        (Join-Path $env:LOCALAPPDATA "com.eloquent.app"),
        (Join-Path $HOME ".LiangLocal")
    )) {
        if (Test-Path -LiteralPath $path) {
            throw "GitHub runner was not clean before installation: $path"
        }
    }

    $freeBefore = Get-FreeDiskGiB
    if ($freeBefore -lt 12) {
        throw "The GitHub runner has only $freeBefore GiB free; at least 12 GiB is required."
    }

    if ($PSCmdlet.ParameterSetName -eq "RemoteInstaller") {
        $downloadStarted = Get-Date
        Invoke-WebRequest -Uri $InstallerUrl -OutFile $installer -UseBasicParsing
        $downloadSeconds = [math]::Round(((Get-Date) - $downloadStarted).TotalSeconds, 1)
    } else {
        $resolvedInstallerPath = (Resolve-Path -LiteralPath $InstallerPath -ErrorAction Stop).ProviderPath
        if (-not (Test-Path -LiteralPath $resolvedInstallerPath -PathType Leaf)) {
            throw "Local installer candidate is not a file: $resolvedInstallerPath"
        }
        $sourceInstallerHash = (Get-FileHash -LiteralPath $resolvedInstallerPath -Algorithm SHA256).Hash
        if (-not $sourceInstallerHash.Equals($InstallerSha256, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Local installer SHA-256 mismatch."
        }
        if ($resolvedInstallerPath.Equals($installer, [StringComparison]::OrdinalIgnoreCase)) {
            $installer = Join-Path $env:RUNNER_TEMP "Mirid-Setup-local-candidate.exe"
        }
        Copy-Item -LiteralPath $resolvedInstallerPath -Destination $installer -Force
    }

    $actualInstallerHash = (Get-FileHash -LiteralPath $installer -Algorithm SHA256).Hash
    if (-not $actualInstallerHash.Equals($InstallerSha256, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Staged installer SHA-256 mismatch."
    }

    $install = Start-Process -FilePath $installer -ArgumentList "/S" -PassThru -Wait
    if ($install.ExitCode -ne 0) {
        throw "Installer exited with code $($install.ExitCode)."
    }
    $installedFile = Get-Item -LiteralPath $installedExecutable -ErrorAction Stop
    if ($installedFile.VersionInfo.ProductVersion -ne $ExpectedVersion) {
        throw "Installed version is $($installedFile.VersionInfo.ProductVersion); expected $ExpectedVersion."
    }
    if (-not (Test-Path -LiteralPath $installerAudioConfig -PathType Leaf)) {
        throw "The installer did not stage its expected first-run audio configuration."
    }
    $unexpectedPrelaunchState = @(
        Get-ChildItem -LiteralPath $appDataRoot -Force |
            Where-Object { $_.FullName -ne $installerAudioConfig }
    )
    if ($unexpectedPrelaunchState.Count -gt 0) {
        $unexpectedNames = $unexpectedPrelaunchState.Name -join ", "
        throw "Mirid created unexpected state before its first launch: $unexpectedNames"
    }
    if (Test-Path -LiteralPath $runtimeRoot) {
        throw "Mirid installed its local runtime before its first launch."
    }
    $prelaunchProcesses = @(
        Get-Process -Name "mirid", "mirid-sidecar-x86_64-pc-windows-msvc", "eloquent-sidecar-x86_64-pc-windows-msvc" -ErrorAction SilentlyContinue
    )
    if ($prelaunchProcesses.Count -gt 0) {
        throw "The silent installer launched Mirid before the clean-install test requested it."
    }

    $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = "--remote-debugging-port=$devToolsPort"
    $env:MIRID_QA_AUTO_BEGIN_SETUP = "1"
    $defaultTtsPortReservation = Reserve-DefaultTtsPort
    $launchStarted = Get-Date
    $process = Start-Process -FilePath $installedExecutable -PassThru
    $env:MIRID_QA_AUTO_BEGIN_SETUP = $previousQaAutoBeginSetup
    $servicePorts = Wait-ForMirid -DesktopProcess $process -DeadlineSeconds $TimeoutSeconds
    $firstLaunchSeconds = [math]::Round(((Get-Date) - $launchStarted).TotalSeconds, 1)
    if ($servicePorts.Backend -ne 8000) {
        throw "Mirid moved the unoccupied backend away from its default port 8000."
    }
    if ($servicePorts.Tts -eq 8002) {
        throw "Mirid did not move TTS away from the deliberately occupied default port 8002."
    }
    $frontendServiceEndpoints = Wait-ForFrontendServiceEndpoints -ExpectedPorts $servicePorts
    $frontendServiceEndpoints |
        ConvertTo-Json -Depth 4 |
        Set-Content -LiteralPath (Join-Path $evidence "frontend-service-endpoints.json") -Encoding utf8
    $serviceListenerProcessIds = @(
        Get-ServiceListenerProcessIds -Ports @($servicePorts.Backend, $servicePorts.Tts)
    )

    $releaseDirectories = @(Get-ChildItem -LiteralPath (Join-Path $runtimeRoot "releases") -Directory -ErrorAction SilentlyContinue)
    $validLayouts = @($releaseDirectories | Where-Object {
        (Test-Path -LiteralPath (Join-Path $_.FullName "mirid-sidecar-x86_64-pc-windows-msvc.exe")) -and
        (Test-Path -LiteralPath (Join-Path $_.FullName "_internal\python312.dll"))
    })
    if ($validLayouts.Count -ne 1) {
        throw "Expected exactly one complete versioned PyInstaller runtime; found $($validLayouts.Count)."
    }
    if (Get-ChildItem -LiteralPath $runtimeRoot -Directory -Filter "_internal-v3-*" -ErrorAction SilentlyContinue) {
        throw "The obsolete renamed _internal layout was created."
    }

    $logText = Get-ChildItem -LiteralPath $logRoot -Filter "*.log" -File |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 |
        Get-Content -Raw
    if ($logText -match "failed to load Python DLL|cannot stage previous runtime|Access is denied \(os error 5\)|Runtime setup failed|winerror 10048|process exited before its local endpoint became ready") {
        throw "The first-launch log contains a release-blocking runtime or local-service error."
    }
    if ($logText -notmatch "Local services are ready\.") {
        throw "The first-launch log does not confirm healthy local services."
    }
    if ($logText -notmatch "Reserved main engine endpoint 127\.0\.0\.1:8000") {
        throw "The first-launch log does not confirm the main engine remained on port 8000."
    }
    if ($logText -notmatch "Voice port 8002 is occupied; Mirid automatically selected") {
        throw "The first-launch log does not confirm automatic TTS fallback from occupied port 8002."
    }

    Stop-MiridNormallyAndVerifyCleanup `
        -DesktopProcess $process `
        -Ports $servicePorts `
        -ListenerProcessIds $serviceListenerProcessIds
    $gracefulShutdownVerified = $true
    $passed = $true
} catch {
    $failure = $_.Exception.Message
    throw
} finally {
    $env:MIRID_QA_AUTO_BEGIN_SETUP = $previousQaAutoBeginSetup
    $env:WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS = $previousBrowserArguments
    Stop-MiridProcesses
    if ($defaultTtsPortReservation) {
        $defaultTtsPortReservation.Stop()
    }
    if (Test-Path -LiteralPath $logRoot) {
        Copy-Item -LiteralPath $logRoot -Destination (Join-Path $evidence "logs") -Recurse -Force -ErrorAction SilentlyContinue
    }
    $videoControllers = @(Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Select-Object Name, AdapterRAM)
    [ordered]@{
        passed = $passed
        failure = $failure
        runner = $env:ImageOS
        expectedVersion = $ExpectedVersion
        installerSource = $PSCmdlet.ParameterSetName
        installerUrl = $InstallerUrl
        installerPath = $InstallerPath
        installerSha256 = $InstallerSha256.ToLowerInvariant()
        downloadSeconds = $downloadSeconds
        firstLaunchSeconds = $firstLaunchSeconds
        freeDiskGiBBefore = if (Get-Variable freeBefore -ErrorAction SilentlyContinue) { $freeBefore } else { $null }
        freeDiskGiBAfter = Get-FreeDiskGiB
        runtimeVersion = if (Test-Path -LiteralPath $runtimeMarker) { (Get-Content -LiteralPath $runtimeMarker -Raw).Trim() } else { "" }
        backendPort = if ($servicePorts) { $servicePorts.Backend } else { $null }
        ttsPort = if ($servicePorts) { $servicePorts.Tts } else { $null }
        frontendBackendPort = if ($frontendServiceEndpoints) { $frontendServiceEndpoints.backendPort } else { $null }
        frontendTtsPort = if ($frontendServiceEndpoints) { $frontendServiceEndpoints.ttsPort } else { $null }
        serviceListenerProcessIds = $serviceListenerProcessIds
        defaultTtsPortOccupiedDuringLaunch = [bool]$defaultTtsPortReservation
        gracefulShutdownReleasedServices = $gracefulShutdownVerified
        videoControllers = $videoControllers
        durationSeconds = [math]::Round(((Get-Date) - $startedAt).TotalSeconds, 1)
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $evidence "summary.json") -Encoding utf8
}
