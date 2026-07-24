use std::collections::VecDeque;
use std::fs;
use std::io::{BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use tauri::{Emitter, Manager};
use tauri_plugin_shell::{process::CommandChild, ShellExt};

#[cfg(all(target_os = "windows", target_arch = "x86_64"))]
include!("runtime_windows.rs");
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
include!("runtime_macos.rs");
#[cfg(not(any(
    all(target_os = "windows", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
compile_error!("Mirid desktop releases currently support Windows x64 and Apple Silicon.");
const LEGACY_APP_ID: &str = "com.eloquent.app";
const MODEL_RUNNER_CONTRACT_VERSION: u64 = 1;

const MAX_RETRIES: u32 = 5;
const INITIAL_BACKOFF_MS: u64 = 2000;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
const DOWNLOAD_ATTEMPT_TIMEOUT: Duration = Duration::from_secs(30 * 60);
const SERVICE_START_TIMEOUT: Duration = Duration::from_secs(5 * 60);
const SERVICE_POLL_INTERVAL: Duration = Duration::from_secs(2);
const PARALLEL_DOWNLOAD_CONNECTIONS: usize = 8;
const PARALLEL_DOWNLOAD_THRESHOLD: u64 = 64 * 1024 * 1024;
const PARALLEL_DOWNLOAD_SEGMENT_SIZE: u64 = 32 * 1024 * 1024;
const DOWNLOAD_PROGRESS_INTERVAL: Duration = Duration::from_secs(1);
const DEFAULT_BACKEND_PORT: u16 = 8000;
const DEFAULT_TTS_PORT: u16 = 8002;
const TTS_PORT_RELEASE_GRACE: Duration = Duration::from_secs(3);

fn destroyed_window_owns_sidecars(window_label: &str) -> bool {
    window_label == "main"
}

#[derive(Clone, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct ServiceEndpoints {
    backend: String,
    secondary: String,
    tts: String,
    backend_port: u16,
    tts_port: u16,
}

struct TtsRuntime {
    port: u16,
    reservation: Option<TcpListener>,
}

struct ServiceRuntime {
    backend_reservation: Mutex<Option<TcpListener>>,
    tts: Mutex<TtsRuntime>,
    tts_operation: Mutex<()>,
}

impl ServiceRuntime {
    fn new() -> Self {
        Self {
            backend_reservation: Mutex::new(None),
            tts: Mutex::new(TtsRuntime {
                port: DEFAULT_TTS_PORT,
                reservation: None,
            }),
            tts_operation: Mutex::new(()),
        }
    }

    fn reserve_initial_ports(&self) -> Result<u16, String> {
        let backend_reservation =
            reserve_fixed_service_port(backend_bind_host(), DEFAULT_BACKEND_PORT)?;
        let (tts_port, tts_reservation) =
            reserve_service_port("127.0.0.1", DEFAULT_TTS_PORT, "voice service")?;
        *self
            .backend_reservation
            .lock()
            .map_err(|_| "backend port reservation is unavailable".to_string())? =
            Some(backend_reservation);
        let mut tts = self
            .tts
            .lock()
            .map_err(|_| "voice-service port state is unavailable".to_string())?;
        tts.port = tts_port;
        tts.reservation = Some(tts_reservation);
        Ok(tts_port)
    }

    fn endpoints(&self) -> Result<ServiceEndpoints, String> {
        let tts_port = self.current_tts_port()?;
        let backend = format!("http://127.0.0.1:{DEFAULT_BACKEND_PORT}");
        Ok(ServiceEndpoints {
            backend: backend.clone(),
            secondary: backend,
            tts: format!("http://127.0.0.1:{tts_port}"),
            backend_port: DEFAULT_BACKEND_PORT,
            tts_port,
        })
    }

    fn current_tts_port(&self) -> Result<u16, String> {
        self.tts
            .lock()
            .map(|state| state.port)
            .map_err(|_| "voice-service port state is unavailable".to_string())
    }

    fn take_backend_reservation(&self) -> Result<TcpListener, String> {
        self.backend_reservation
            .lock()
            .map_err(|_| "backend port reservation is unavailable".to_string())?
            .take()
            .ok_or_else(|| "backend port 8000 is not reserved".to_string())
    }

    fn take_tts_reservation(&self) -> Result<(u16, TcpListener), String> {
        let mut state = self
            .tts
            .lock()
            .map_err(|_| "voice-service port state is unavailable".to_string())?;
        let reservation = state
            .reservation
            .take()
            .ok_or_else(|| "voice-service port is not reserved".to_string())?;
        Ok((state.port, reservation))
    }

    fn reserve_tts_for_restart(
        &self,
        preferred_port: u16,
        release_grace: Duration,
    ) -> Result<u16, String> {
        let (port, reservation) = reserve_service_port_after_release(
            "127.0.0.1",
            preferred_port,
            "voice service",
            release_grace,
        )?;
        let mut state = self
            .tts
            .lock()
            .map_err(|_| "voice-service port state is unavailable".to_string())?;
        state.port = port;
        state.reservation = Some(reservation);
        Ok(port)
    }
}

fn reserve_fixed_service_port(host: &str, port: u16) -> Result<TcpListener, String> {
    reserve_fixed_service_port_after_release(host, port, Duration::ZERO)
}

fn reserve_fixed_service_port_after_release(
    host: &str,
    port: u16,
    release_grace: Duration,
) -> Result<TcpListener, String> {
    let started = Instant::now();
    loop {
        match TcpListener::bind((host, port)) {
            Ok(listener) => return Ok(listener),
            Err(_) if started.elapsed() < release_grace => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(error) => {
                return Err(format!(
                    "Main engine port {port} is already in use or unavailable. Close the other program or Mirid session using port {port}, then reopen Mirid. ({error})"
                ));
            }
        }
    }
}

fn reserve_service_port(
    host: &str,
    preferred_port: u16,
    label: &str,
) -> Result<(u16, TcpListener), String> {
    reserve_service_port_after_release(host, preferred_port, label, Duration::ZERO)
}

fn reserve_service_port_after_release(
    host: &str,
    preferred_port: u16,
    label: &str,
    release_grace: Duration,
) -> Result<(u16, TcpListener), String> {
    let started = Instant::now();
    let preferred_error = loop {
        match TcpListener::bind((host, preferred_port)) {
            Ok(listener) => return Ok((preferred_port, listener)),
            Err(_) if started.elapsed() < release_grace => {
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(error) => break error,
        }
    };

    let listener = TcpListener::bind((host, 0)).map_err(|fallback_error| {
        format!(
            "cannot reserve a local port for {label}: preferred port {preferred_port} is unavailable ({preferred_error}); automatic fallback also failed ({fallback_error})"
        )
    })?;
    let port = listener
        .local_addr()
        .map_err(|error| format!("cannot read the reserved {label} port: {error}"))?
        .port();
    Ok((port, listener))
}

#[cfg(target_os = "windows")]
struct SidecarJob {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

#[cfg(target_os = "windows")]
unsafe impl Send for SidecarJob {}
#[cfg(target_os = "windows")]
unsafe impl Sync for SidecarJob {}

#[cfg(target_os = "windows")]
impl SidecarJob {
    fn attach(process_id: u32) -> Result<Self, String> {
        use std::mem::size_of;
        use std::ptr::{null, null_mut};
        use windows_sys::Win32::Foundation::{CloseHandle, FALSE};
        use windows_sys::Win32::System::JobObjects::{
            AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
            SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
            JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        };
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
        };

        unsafe {
            let job = CreateJobObjectW(null(), null());
            if job.is_null() {
                return Err(format!(
                    "cannot create a Windows sidecar job: {}",
                    std::io::Error::last_os_error()
                ));
            }

            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &limits as *const _ as *const core::ffi::c_void,
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) == FALSE
            {
                let error = std::io::Error::last_os_error();
                CloseHandle(job);
                return Err(format!("cannot configure the Windows sidecar job: {error}"));
            }

            let process = OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, FALSE, process_id);
            if process == null_mut() {
                let error = std::io::Error::last_os_error();
                CloseHandle(job);
                return Err(format!(
                    "cannot open sidecar process {process_id} for containment: {error}"
                ));
            }
            let assigned = AssignProcessToJobObject(job, process);
            CloseHandle(process);
            if assigned == FALSE {
                let error = std::io::Error::last_os_error();
                CloseHandle(job);
                return Err(format!(
                    "cannot contain sidecar process {process_id}: {error}"
                ));
            }

            Ok(Self { handle: job })
        }
    }

    fn terminate(&self) -> Result<(), String> {
        use windows_sys::Win32::Foundation::FALSE;
        use windows_sys::Win32::System::JobObjects::TerminateJobObject;

        if unsafe { TerminateJobObject(self.handle, 1) } == FALSE {
            return Err(format!(
                "cannot terminate the Windows sidecar job: {}",
                std::io::Error::last_os_error()
            ));
        }
        Ok(())
    }

    fn contains_process(&self, process_id: u32) -> bool {
        use windows_sys::Win32::Foundation::{CloseHandle, FALSE};
        use windows_sys::Win32::System::JobObjects::IsProcessInJob;
        use windows_sys::Win32::System::Threading::{
            OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION,
        };

        unsafe {
            let process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, process_id);
            if process.is_null() {
                return false;
            }
            let mut belongs = FALSE;
            let checked = IsProcessInJob(process, self.handle, &mut belongs);
            CloseHandle(process);
            checked != FALSE && belongs != FALSE
        }
    }
}

#[cfg(target_os = "windows")]
impl Drop for SidecarJob {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.handle);
        }
    }
}

struct ManagedSidecar {
    child: CommandChild,
    exited: Arc<AtomicBool>,
    #[cfg(target_os = "windows")]
    job: Option<SidecarJob>,
}

impl ManagedSidecar {
    fn new(child: CommandChild) -> Self {
        #[cfg(target_os = "windows")]
        let job = match SidecarJob::attach(child.pid()) {
            Ok(job) => Some(job),
            Err(error) => {
                log::warn!(
                    "Could not place sidecar process {} in a Windows Job Object: {error}",
                    child.pid()
                );
                None
            }
        };
        Self {
            child,
            exited: Arc::new(AtomicBool::new(false)),
            #[cfg(target_os = "windows")]
            job,
        }
    }

    fn pid(&self) -> u32 {
        self.child.pid()
    }

    fn exit_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.exited)
    }

    fn is_running(&self) -> bool {
        !self.exited.load(Ordering::Acquire)
    }

    #[cfg(target_os = "windows")]
    fn owns_process(&self, process_id: u32) -> bool {
        process_id == self.child.pid()
            || self
                .job
                .as_ref()
                .map(|job| job.contains_process(process_id))
                .unwrap_or(false)
    }

    fn kill(self) -> Result<(), String> {
        #[cfg(target_os = "windows")]
        if let Some(job) = &self.job {
            if job.terminate().is_ok() {
                return Ok(());
            }
        }
        #[cfg(target_os = "windows")]
        if terminate_windows_process_tree(self.child.pid()).is_ok() {
            return Ok(());
        }
        self.child.kill().map_err(|error| error.to_string())
    }
}

#[cfg(target_os = "windows")]
fn windows_listener_process_ids(port: u16) -> Result<Vec<u32>, String> {
    use std::mem::size_of;
    use std::ptr::{null_mut, read_unaligned};
    use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, FALSE};
    use windows_sys::Win32::NetworkManagement::IpHelper::{
        GetExtendedTcpTable, MIB_TCPROW_OWNER_PID, TCP_TABLE_OWNER_PID_LISTENER,
    };

    const AF_INET: u32 = 2;
    let mut table_size = 0u32;
    let initial_status = unsafe {
        GetExtendedTcpTable(
            null_mut(),
            &mut table_size,
            FALSE,
            AF_INET,
            TCP_TABLE_OWNER_PID_LISTENER,
            0,
        )
    };
    if initial_status != ERROR_INSUFFICIENT_BUFFER && initial_status != 0 {
        return Err(format!(
            "cannot inspect local TCP listener ownership (Windows error {initial_status})"
        ));
    }
    if table_size == 0 {
        return Ok(Vec::new());
    }

    let mut table = vec![0u8; table_size as usize];
    let status = unsafe {
        GetExtendedTcpTable(
            table.as_mut_ptr().cast(),
            &mut table_size,
            FALSE,
            AF_INET,
            TCP_TABLE_OWNER_PID_LISTENER,
            0,
        )
    };
    if status != 0 {
        return Err(format!(
            "cannot read local TCP listener ownership (Windows error {status})"
        ));
    }

    let count = unsafe { read_unaligned(table.as_ptr().cast::<u32>()) } as usize;
    let row_size = size_of::<MIB_TCPROW_OWNER_PID>();
    let rows_size = count
        .checked_mul(row_size)
        .and_then(|size| size.checked_add(size_of::<u32>()))
        .ok_or_else(|| "local TCP listener table size overflowed".to_string())?;
    if rows_size > table.len() {
        return Err("Windows returned an incomplete TCP listener table".to_string());
    }

    let mut process_ids = Vec::new();
    for index in 0..count {
        let offset = size_of::<u32>() + index * row_size;
        let row =
            unsafe { read_unaligned(table.as_ptr().add(offset).cast::<MIB_TCPROW_OWNER_PID>()) };
        if u16::from_be(row.dwLocalPort as u16) == port {
            process_ids.push(row.dwOwningPid);
        }
    }
    Ok(process_ids)
}

#[cfg(target_os = "windows")]
fn terminate_windows_process_tree(process_id: u32) -> Result<(), String> {
    use std::os::windows::process::CommandExt;
    use std::process::{Command, Stdio};

    const CREATE_NO_WINDOW: u32 = 0x08000000;
    let status = Command::new("taskkill.exe")
        .args(["/PID", &process_id.to_string(), "/T", "/F"])
        .creation_flags(CREATE_NO_WINDOW)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|error| format!("cannot launch taskkill for sidecar {process_id}: {error}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "taskkill could not terminate sidecar process tree {process_id}"
        ))
    }
}

#[derive(Default)]
struct Sidecars {
    backend: Mutex<Option<ManagedSidecar>>,
    tts: Mutex<Option<ManagedSidecar>>,
}

#[derive(Default)]
struct RuntimeBootState(Mutex<Option<BootProgress>>);

#[derive(Default)]
struct RuntimeSetupGate {
    allowed: Mutex<bool>,
    ready: Condvar,
}

#[derive(serde::Serialize)]
struct SidecarStatus {
    backend: bool,
    tts: bool,
}

#[derive(Clone, serde::Serialize)]
struct BootProgress {
    stage: String,
    message: String,
    percent: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    downloaded_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes_per_second: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    eta_seconds: Option<u64>,
}

fn extraction_percent(extracted_bytes: u64, total_bytes: u64) -> u8 {
    if total_bytes == 0 {
        return 0;
    }
    ((extracted_bytes.saturating_mul(100) / total_bytes).min(99)) as u8
}

fn format_runtime_size(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    format!("{:.1} GB", bytes as f64 / GIB)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DownloadSnapshot {
    downloaded_bytes: u64,
    total_bytes: u64,
    bytes_per_second: Option<u64>,
    eta_seconds: Option<u64>,
    percent: u8,
}

struct DownloadTelemetry {
    last_emit_at: Instant,
    last_sample_at: Instant,
    last_sample_bytes: u64,
    smoothed_bytes_per_second: f64,
    has_rate_sample: bool,
}

impl DownloadTelemetry {
    fn new(initial_bytes: u64) -> Self {
        let now = Instant::now();
        Self {
            last_emit_at: now.checked_sub(DOWNLOAD_PROGRESS_INTERVAL).unwrap_or(now),
            last_sample_at: now,
            last_sample_bytes: initial_bytes,
            smoothed_bytes_per_second: 0.0,
            has_rate_sample: false,
        }
    }

    fn observe(
        &mut self,
        downloaded_bytes: u64,
        total_bytes: u64,
        force: bool,
    ) -> Option<DownloadSnapshot> {
        let now = Instant::now();
        let sample_elapsed = now.saturating_duration_since(self.last_sample_at);
        if sample_elapsed >= DOWNLOAD_PROGRESS_INTERVAL {
            let transferred = downloaded_bytes.saturating_sub(self.last_sample_bytes);
            let instant_rate = transferred as f64 / sample_elapsed.as_secs_f64();
            self.smoothed_bytes_per_second = if self.has_rate_sample {
                (self.smoothed_bytes_per_second * 0.7) + (instant_rate * 0.3)
            } else {
                instant_rate
            };
            self.has_rate_sample = true;
            self.last_sample_at = now;
            self.last_sample_bytes = downloaded_bytes;
        }

        if !force && now.saturating_duration_since(self.last_emit_at) < DOWNLOAD_PROGRESS_INTERVAL {
            return None;
        }
        self.last_emit_at = now;

        let bytes_per_second = self
            .has_rate_sample
            .then_some(self.smoothed_bytes_per_second.max(0.0).round() as u64);
        let eta_seconds = bytes_per_second
            .filter(|speed| *speed > 0)
            .map(|speed| total_bytes.saturating_sub(downloaded_bytes).div_ceil(speed));
        let percent = if total_bytes == 0 {
            0
        } else {
            ((downloaded_bytes.saturating_mul(100)) / total_bytes).min(100) as u8
        };

        Some(DownloadSnapshot {
            downloaded_bytes,
            total_bytes,
            bytes_per_second,
            eta_seconds,
            percent,
        })
    }
}

struct ExtractionProgressReader<'a> {
    inner: &'a mut dyn Read,
    app: &'a tauri::AppHandle,
    extracted_bytes: &'a mut u64,
    total_bytes: u64,
    completed_entries: usize,
    total_entries: usize,
    last_percent: &'a mut u8,
}

impl Read for ExtractionProgressReader<'_> {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        let count = self.inner.read(buffer)?;
        if count == 0 {
            return Ok(0);
        }

        *self.extracted_bytes = (*self.extracted_bytes).saturating_add(count as u64);
        let percent = extraction_percent(*self.extracted_bytes, self.total_bytes);
        if percent > *self.last_percent {
            *self.last_percent = percent;
            emit(
                self.app,
                "extract",
                &format!(
                    "Installing runtime... {} of {} files · {} of {}",
                    self.completed_entries,
                    self.total_entries,
                    format_runtime_size(*self.extracted_bytes),
                    format_runtime_size(self.total_bytes),
                ),
                percent,
            );
        }
        Ok(count)
    }
}

fn emit_progress(app: &tauri::AppHandle, progress: BootProgress) {
    if let Some(state) = app.try_state::<RuntimeBootState>() {
        if let Ok(mut current) = state.0.lock() {
            *current = Some(progress.clone());
        }
    }
    let _ = app.emit("runtime-boot", progress);
}

fn emit(app: &tauri::AppHandle, stage: &str, message: &str, percent: u8) {
    emit_progress(
        app,
        BootProgress {
            stage: stage.to_string(),
            message: message.to_string(),
            percent,
            downloaded_bytes: None,
            total_bytes: None,
            bytes_per_second: None,
            eta_seconds: None,
        },
    );
    log::info!("[boot] {stage} ({percent}%): {message}");
}

fn emit_download(app: &tauri::AppHandle, stage: &str, message: &str, snapshot: DownloadSnapshot) {
    emit_progress(
        app,
        BootProgress {
            stage: stage.to_string(),
            message: message.to_string(),
            percent: snapshot.percent,
            downloaded_bytes: Some(snapshot.downloaded_bytes),
            total_bytes: Some(snapshot.total_bytes),
            bytes_per_second: snapshot.bytes_per_second,
            eta_seconds: snapshot.eta_seconds,
        },
    );
    log::info!(
        "[boot] {stage} ({}%): {message} · {} B/s",
        snapshot.percent,
        snapshot.bytes_per_second.unwrap_or(0)
    );
}

#[tauri::command]
fn get_runtime_boot_status(
    state: tauri::State<'_, RuntimeBootState>,
) -> Result<Option<BootProgress>, String> {
    state
        .0
        .lock()
        .map(|current| current.clone())
        .map_err(|_| "runtime boot state is unavailable".to_string())
}

#[tauri::command]
fn begin_runtime_setup(state: tauri::State<'_, RuntimeSetupGate>) -> Result<(), String> {
    let mut allowed = state
        .allowed
        .lock()
        .map_err(|_| "runtime setup gate is unavailable".to_string())?;
    *allowed = true;
    state.ready.notify_all();
    Ok(())
}

fn wait_for_runtime_setup(app: &tauri::AppHandle) -> Result<(), String> {
    let qa_auto_begin = std::env::var("MIRID_QA_AUTO_BEGIN_SETUP")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        })
        .unwrap_or(false);
    if runtime_is_ready(app) || qa_auto_begin {
        return Ok(());
    }
    emit(
        app,
        "awaiting_setup",
        "Choose how you want to begin before Mirid installs its local engine.",
        0,
    );
    let gate = app.state::<RuntimeSetupGate>();
    let mut allowed = gate
        .allowed
        .lock()
        .map_err(|_| "runtime setup gate is unavailable".to_string())?;
    while !*allowed {
        allowed = gate
            .ready
            .wait(allowed)
            .map_err(|_| "runtime setup gate is unavailable".to_string())?;
    }
    Ok(())
}

fn runtime_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let base = app
        .path()
        .app_local_data_dir()
        .map_err(|e| format!("cannot resolve app data dir: {e}"))?;
    Ok(base.join("runtime"))
}

fn user_data_dir(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    let base = app
        .path()
        .app_local_data_dir()
        .map_err(|e| format!("cannot resolve app data dir: {e}"))?;
    Ok(base.join("data"))
}

fn copy_dir_contents(source: &Path, destination: &Path, overwrite: bool) -> Result<(), String> {
    if !source.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(destination).map_err(|error| error.to_string())?;
    for entry in fs::read_dir(source).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        if file_type.is_symlink() {
            continue;
        }
        let target = destination.join(entry.file_name());
        if file_type.is_dir() {
            copy_dir_contents(&entry.path(), &target, overwrite)?;
        } else if file_type.is_file() && (overwrite || !target.exists()) {
            fs::copy(entry.path(), &target).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn is_managed_avatar_filename(filename: &str) -> bool {
    let Some((stem, extension)) = filename.rsplit_once('.') else {
        return false;
    };
    if !matches!(
        extension.to_ascii_lowercase().as_str(),
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "mp4" | "webm" | "mov" | "m4v"
    ) {
        return false;
    }
    if stem.len() != 36 {
        return false;
    }
    stem.as_bytes().iter().enumerate().all(|(index, value)| {
        if matches!(index, 8 | 13 | 18 | 23) {
            *value == b'-'
        } else {
            value.is_ascii_hexdigit()
        }
    })
}

fn copy_runtime_avatar_files(source: &Path, destination: &Path) -> Result<(), String> {
    if !source.is_dir() {
        return Ok(());
    }
    fs::create_dir_all(destination).map_err(|error| error.to_string())?;
    for entry in fs::read_dir(source).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let file_type = entry.file_type().map_err(|error| error.to_string())?;
        if !file_type.is_file() || file_type.is_symlink() {
            continue;
        }
        let filename = entry.file_name();
        let Some(filename_text) = filename.to_str() else {
            continue;
        };
        if !is_managed_avatar_filename(filename_text) {
            continue;
        }
        let target = destination.join(filename);
        if !target.exists() {
            fs::copy(entry.path(), target).map_err(|error| error.to_string())?;
        }
    }
    Ok(())
}

fn migrate_runtime_user_data_from_paths(runtime: &Path, destination: &Path) -> Result<(), String> {
    fs::create_dir_all(&destination).map_err(|error| error.to_string())?;
    let installed = installed_runtime_internal_dirs(runtime);
    if let Some(internal) = installed.last() {
        for source in [
            internal.join("backend").join("data"),
            internal.join("backend").join("app").join("data"),
        ] {
            copy_dir_contents(&source, &destination, false)?;
        }
        copy_dir_contents(
            &internal
                .join("backend")
                .join("app")
                .join("static")
                .join("documents"),
            &destination.join("documents"),
            false,
        )?;
        copy_dir_contents(
            &internal.join("backend").join("voxcpm_gguf_models"),
            &destination.join("models").join("voxcpm_gguf"),
            false,
        )?;
        copy_dir_contents(&internal.join("logs"), &destination.join("logs"), false)?;
    }
    for internal in installed.into_iter().rev() {
        copy_runtime_avatar_files(
            &internal.join("backend").join("app").join("static"),
            &destination.join("avatars"),
        )?;
    }
    Ok(())
}

fn migrate_runtime_user_data(app: &tauri::AppHandle) -> Result<(), String> {
    let runtime = runtime_dir(app)?;
    let destination = user_data_dir(app)?;
    migrate_runtime_user_data_from_paths(&runtime, &destination)
}

fn preserve_runtime_static_data(current: &Path, staging: &Path) -> Result<(), String> {
    for relative in [
        Path::new("backend/app/static/voice_references"),
        Path::new("backend/app/static/generated_images"),
        Path::new("backend/app/static/room_gallery"),
        Path::new("backend/app/static/outreach_runtime"),
        Path::new("backend/static/voice_references"),
    ] {
        copy_dir_contents(&current.join(relative), &staging.join(relative), true)?;
    }
    Ok(())
}

fn backend_host_from_settings(settings: &serde_json::Value) -> &'static str {
    let lan_enabled = settings
        .get("openaiServerLanEnabled")
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let has_password = settings
        .get("admin_password")
        .and_then(serde_json::Value::as_str)
        .is_some_and(|password| !password.trim().is_empty());
    if lan_enabled && has_password {
        "0.0.0.0"
    } else {
        "127.0.0.1"
    }
}

fn backend_bind_host() -> &'static str {
    let Some(home) = std::env::var_os("USERPROFILE").map(PathBuf::from) else {
        return "127.0.0.1";
    };
    let settings_path = home.join(".LiangLocal").join("settings.json");
    let Ok(contents) = fs::read_to_string(settings_path) else {
        return "127.0.0.1";
    };
    let Ok(settings) = serde_json::from_str::<serde_json::Value>(&contents) else {
        return "127.0.0.1";
    };
    backend_host_from_settings(&settings)
}

fn migrate_legacy_runtime(app: &tauri::AppHandle) -> Result<(), String> {
    let destination = runtime_dir(app)?;
    if destination.exists() {
        return Ok(());
    }
    let current_app_dir = destination
        .parent()
        .ok_or_else(|| "cannot resolve Mirid app data directory".to_string())?;
    let app_data_root = current_app_dir
        .parent()
        .ok_or_else(|| "cannot resolve the application data root".to_string())?;
    let legacy = app_data_root.join(LEGACY_APP_ID).join("runtime");
    if !legacy.exists() {
        return Ok(());
    }
    let parent = destination
        .parent()
        .ok_or_else(|| "cannot resolve Mirid app data directory".to_string())?;
    fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    fs::rename(&legacy, &destination).map_err(|e| format!("cannot migrate legacy runtime: {e}"))?;
    Ok(())
}

fn legacy_runtime_internal_dir(runtime: &Path) -> PathBuf {
    runtime.join("_internal")
}

fn versioned_runtime_release_dir(runtime: &Path) -> PathBuf {
    runtime.join("releases").join(format!(
        "{RUNTIME_VERSION}-{}-{}",
        &RUNTIME_ARCHIVE_SHA256[..12],
        &SIDECAR_EXE_SHA256[..12]
    ))
}

fn versioned_runtime_internal_dir(runtime: &Path) -> PathBuf {
    versioned_runtime_release_dir(runtime).join("_internal")
}

fn legacy_sidecar_exe_path(runtime: &Path) -> PathBuf {
    runtime.join(SIDECAR_EXE)
}

fn versioned_sidecar_exe_path(runtime: &Path) -> PathBuf {
    versioned_runtime_release_dir(runtime).join(SIDECAR_EXE)
}

#[derive(Debug, PartialEq, Eq)]
struct RuntimeLayout {
    internal: PathBuf,
    sidecar: PathBuf,
}

fn runtime_internal_is_complete(internal: &Path) -> bool {
    if !internal.join("backend").is_dir() {
        return false;
    }
    #[cfg(target_os = "windows")]
    if !internal.join("python312.dll").is_file() {
        return false;
    }
    true
}

fn installed_runtime_internal_dirs(runtime: &Path) -> Vec<PathBuf> {
    let releases = runtime.join("releases");
    let mut candidates = fs::read_dir(releases)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false))
        .map(|entry| entry.path().join("_internal"))
        .filter(|internal| runtime_internal_is_complete(internal))
        .collect::<Vec<_>>();
    let legacy = legacy_runtime_internal_dir(runtime);
    if runtime_internal_is_complete(&legacy) {
        candidates.push(legacy);
    }
    candidates.sort_by(|left, right| {
        runtime_internal_generation(left)
            .cmp(&runtime_internal_generation(right))
            .then_with(|| left.cmp(right))
    });
    candidates.dedup();
    candidates
}

fn runtime_internal_generation(internal: &Path) -> u64 {
    internal
        .parent()
        .and_then(Path::file_name)
        .and_then(|name| name.to_str())
        .and_then(|name| name.strip_prefix('v'))
        .and_then(|name| name.split('-').next())
        .and_then(|generation| generation.parse().ok())
        .unwrap_or(0)
}

fn existing_runtime_internal_dir(runtime: &Path) -> Option<PathBuf> {
    installed_runtime_internal_dirs(runtime).pop()
}

fn active_runtime_layout(runtime: &Path) -> Option<RuntimeLayout> {
    let internal = versioned_runtime_internal_dir(runtime);
    let sidecar = versioned_sidecar_exe_path(runtime);
    if runtime_internal_is_complete(&internal) && sidecar.is_file() {
        return Some(RuntimeLayout { internal, sidecar });
    }
    let legacy = RuntimeLayout {
        internal: legacy_runtime_internal_dir(runtime),
        sidecar: legacy_sidecar_exe_path(runtime),
    };
    (runtime_internal_is_complete(&legacy.internal) && legacy.sidecar.is_file()).then_some(legacy)
}

fn runtime_assets_are_reusable(
    internal: &Path,
    sidecar: &Path,
    expected_sidecar_size: u64,
    expected_sidecar_sha256: &str,
) -> Result<bool, String> {
    if !runtime_internal_is_complete(internal) {
        return Ok(false);
    }
    file_matches(sidecar, expected_sidecar_size, expected_sidecar_sha256)
}

fn ready_marker(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    Ok(runtime_dir(app)?.join("runtime.ready"))
}

fn runtime_is_ready(app: &tauri::AppHandle) -> bool {
    let runtime = match runtime_dir(app) {
        Ok(directory) => directory,
        Err(_) => return false,
    };
    let marker = match ready_marker(app) {
        Ok(m) => m,
        Err(_) => return false,
    };
    if fs::read_to_string(&marker)
        .map(|version| version.trim() != RUNTIME_VERSION)
        .unwrap_or(true)
    {
        return false;
    }
    active_runtime_layout(&runtime).is_some()
}

fn sha256_file(path: &Path) -> Result<String, String> {
    let mut file =
        fs::File::open(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn file_matches(path: &Path, expected_size: u64, expected_sha256: &str) -> Result<bool, String> {
    if !path.exists() {
        return Ok(false);
    }
    let actual_size = fs::metadata(path).map_err(|e| e.to_string())?.len();
    if actual_size != expected_size {
        log::warn!(
            "{} has the wrong size ({} instead of {})",
            path.display(),
            actual_size,
            expected_size
        );
        return Ok(false);
    }
    let actual_sha256 = sha256_file(path)?;
    if !actual_sha256.eq_ignore_ascii_case(expected_sha256) {
        log::warn!("{} failed SHA-256 verification", path.display());
        return Ok(false);
    }
    Ok(true)
}

fn promote_download(partial: &Path, destination: &Path) -> Result<(), String> {
    if destination.exists() {
        fs::remove_file(destination)
            .map_err(|e| format!("cannot replace {}: {e}", destination.display()))?;
    }
    fs::rename(partial, destination)
        .map_err(|e| format!("cannot promote {}: {e}", destination.display()))
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut permissions = fs::metadata(path)
        .map_err(|error| error.to_string())?
        .permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(path, permissions).map_err(|error| error.to_string())
}

#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn content_range_starts_at(
    value: Option<&reqwest::header::HeaderValue>,
    expected_start: u64,
) -> bool {
    let Some(value) = value.and_then(|header| header.to_str().ok()) else {
        return false;
    };
    value.starts_with(&format!("bytes {expected_start}-"))
}

fn archive_path_is_safe(path: &str) -> bool {
    !Path::new(path).components().any(|component| {
        matches!(
            component,
            Component::ParentDir | Component::RootDir | Component::Prefix(_)
        )
    })
}

enum DownloadMessage {
    Bytes(u64),
    Finished(Result<(), String>),
}

fn path_with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

fn runtime_attempt_path(dest: &Path, label: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    path_with_suffix(dest, &format!(".{label}-{}-{nonce}", std::process::id()))
}

fn download_chunk_path(partial: &Path, index: usize) -> PathBuf {
    path_with_suffix(partial, &format!(".chunk-{index}"))
}

fn remove_download_chunks(partial: &Path) {
    let Some(parent) = partial.parent() else {
        return;
    };
    let Some(filename) = partial.file_name().and_then(|value| value.to_str()) else {
        return;
    };
    let prefix = format!("{filename}.chunk-");
    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            if name.to_string_lossy().starts_with(&prefix) {
                let _ = fs::remove_file(entry.path());
            }
        }
    }
    let _ = fs::remove_file(path_with_suffix(partial, ".assembling"));
}

fn cleanup_download_artifacts(runtime: &Path) {
    for destination in [
        runtime.join(RUNTIME_ARCHIVE),
        legacy_sidecar_exe_path(runtime),
        versioned_sidecar_exe_path(runtime),
    ] {
        let partial = destination.with_extension("part");
        remove_download_chunks(&partial);
        let _ = fs::remove_file(partial);
    }
}

fn cleanup_runtime_staging_artifacts(internal: &Path) {
    let Some(parent) = internal.parent() else {
        return;
    };
    let Some(filename) = internal.file_name().and_then(|value| value.to_str()) else {
        return;
    };
    let legacy_name = format!("{filename}.installing");
    let attempt_prefix = format!("{legacy_name}-");

    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name == legacy_name || name.starts_with(&attempt_prefix) {
                if let Err(error) = fs::remove_dir_all(entry.path()) {
                    log::warn!(
                        "Could not remove abandoned runtime staging directory {}: {error}",
                        entry.path().display()
                    );
                }
            }
        }
    }
}

fn download_ranges(expected_size: u64, segment_size: u64) -> Vec<(u64, u64)> {
    if expected_size == 0 {
        return Vec::new();
    }
    let segment_size = segment_size.max(1);
    (0..expected_size.div_ceil(segment_size))
        .map(|index| {
            let start = index * segment_size;
            (start, (start + segment_size - 1).min(expected_size - 1))
        })
        .collect()
}

fn download_range(
    client: reqwest::blocking::Client,
    url: String,
    path: PathBuf,
    range_start: u64,
    range_end: u64,
    progress: std::sync::mpsc::Sender<DownloadMessage>,
) -> Result<(), String> {
    let expected_length = range_end - range_start + 1;
    for attempt in 1..=MAX_RETRIES {
        let mut existing = path.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        if existing > expected_length {
            fs::remove_file(&path).map_err(|error| error.to_string())?;
            existing = 0;
        }
        if existing == expected_length {
            return Ok(());
        }

        let request_start = range_start + existing;
        let response = client
            .get(&url)
            .header(
                reqwest::header::RANGE,
                format!("bytes={request_start}-{range_end}"),
            )
            .send();
        let mut response = match response {
            Ok(response) => response,
            Err(error) => {
                if attempt == MAX_RETRIES {
                    return Err(format!("range request failed: {error}"));
                }
                std::thread::sleep(Duration::from_millis(
                    INITIAL_BACKOFF_MS * 2_u64.pow(attempt - 1),
                ));
                continue;
            }
        };

        if response.status() != reqwest::StatusCode::PARTIAL_CONTENT
            || !content_range_starts_at(
                response.headers().get(reqwest::header::CONTENT_RANGE),
                request_start,
            )
        {
            return Err(format!(
                "server did not honour byte range {request_start}-{range_end}"
            ));
        }

        let expected_remaining = expected_length - existing;
        if response
            .content_length()
            .is_some_and(|length| length != expected_remaining)
        {
            return Err(format!(
                "range response length did not match {request_start}-{range_end}"
            ));
        }

        let mut file = if existing > 0 {
            fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .map_err(|error| error.to_string())?
        } else {
            fs::File::create(&path).map_err(|error| error.to_string())?
        };
        let mut buffer = [0u8; 1024 * 1024];
        let mut read_failed = None;
        loop {
            match response.read(&mut buffer) {
                Ok(0) => break,
                Ok(read) => {
                    file.write_all(&buffer[..read])
                        .map_err(|error| error.to_string())?;
                    existing += read as u64;
                    let _ = progress.send(DownloadMessage::Bytes(read as u64));
                }
                Err(error) => {
                    read_failed = Some(error.to_string());
                    break;
                }
            }
        }
        file.flush().map_err(|error| error.to_string())?;

        if existing == expected_length {
            return Ok(());
        }
        if existing > expected_length {
            return Err("range download exceeded its expected length".to_string());
        }
        if attempt == MAX_RETRIES {
            return Err(format!(
                "range download stopped at {existing}/{expected_length} bytes{}",
                read_failed
                    .map(|error| format!(": {error}"))
                    .unwrap_or_default()
            ));
        }
        std::thread::sleep(Duration::from_millis(
            INITIAL_BACKOFF_MS * 2_u64.pow(attempt - 1),
        ));
    }
    Err("range download failed".to_string())
}

fn download_file_parallel<F>(
    client: &reqwest::blocking::Client,
    url: &str,
    partial: &Path,
    expected_size: u64,
    connection_count: usize,
    segment_size: u64,
    mut progress: F,
) -> Result<(), String>
where
    F: FnMut(u64),
{
    let ranges = download_ranges(expected_size, segment_size);
    if ranges.is_empty() {
        return Err("download has no byte ranges".to_string());
    }
    let worker_count = connection_count.max(1).min(ranges.len());
    let chunk_paths = (0..ranges.len())
        .map(|index| download_chunk_path(partial, index))
        .collect::<Vec<_>>();

    if partial.exists() {
        let partial_length = partial.metadata().map_err(|error| error.to_string())?.len();
        let first_chunk_length = ranges[0].1 - ranges[0].0 + 1;
        if !chunk_paths[0].exists() && partial_length <= first_chunk_length {
            fs::rename(partial, &chunk_paths[0]).map_err(|error| error.to_string())?;
        } else {
            fs::remove_file(partial).map_err(|error| error.to_string())?;
        }
    }

    let mut downloaded = 0u64;
    for (path, (start, end)) in chunk_paths.iter().zip(ranges.iter()) {
        let expected_length = end - start + 1;
        let length = path.metadata().map(|metadata| metadata.len()).unwrap_or(0);
        if length > expected_length {
            fs::remove_file(path).map_err(|error| error.to_string())?;
        } else {
            downloaded += length;
        }
    }
    progress(downloaded);

    let ranges = Arc::new(ranges);
    let chunk_paths = Arc::new(chunk_paths);
    let pending = Arc::new(Mutex::new(VecDeque::from_iter(0..ranges.len())));
    let (sender, receiver) = std::sync::mpsc::channel();
    let mut workers = Vec::with_capacity(worker_count);
    for _ in 0..worker_count {
        let client = client.clone();
        let url = url.to_string();
        let sender = sender.clone();
        let ranges = Arc::clone(&ranges);
        let chunk_paths = Arc::clone(&chunk_paths);
        let pending = Arc::clone(&pending);
        workers.push(std::thread::spawn(move || loop {
            let index = match pending.lock() {
                Ok(mut queue) => queue.pop_front(),
                Err(_) => {
                    let _ = sender.send(DownloadMessage::Finished(Err(
                        "download queue became unavailable".to_string(),
                    )));
                    return;
                }
            };
            let Some(index) = index else {
                let _ = sender.send(DownloadMessage::Finished(Ok(())));
                return;
            };
            let (start, end) = ranges[index];
            let result = download_range(
                client.clone(),
                url.clone(),
                chunk_paths[index].clone(),
                start,
                end,
                sender.clone(),
            );
            if let Err(error) = result {
                let _ = sender.send(DownloadMessage::Finished(Err(error)));
                return;
            }
        }));
    }
    drop(sender);

    let mut completed = 0usize;
    let mut errors = Vec::new();
    while completed < workers.len() {
        match receiver.recv_timeout(DOWNLOAD_PROGRESS_INTERVAL) {
            Ok(DownloadMessage::Bytes(bytes)) => {
                downloaded += bytes;
                progress(downloaded.min(expected_size));
            }
            Ok(DownloadMessage::Finished(result)) => {
                completed += 1;
                if let Err(error) = result {
                    errors.push(error);
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => progress(downloaded),
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    for worker in workers {
        if worker.join().is_err() {
            errors.push("download worker stopped unexpectedly".to_string());
        }
    }
    if !errors.is_empty() {
        return Err(errors.join("; "));
    }

    let assembling = path_with_suffix(partial, ".assembling");
    let _ = fs::remove_file(&assembling);
    let mut output = fs::File::create(&assembling).map_err(|error| error.to_string())?;
    for path in chunk_paths.iter() {
        let mut input = fs::File::open(path).map_err(|error| error.to_string())?;
        std::io::copy(&mut input, &mut output).map_err(|error| error.to_string())?;
    }
    output.flush().map_err(|error| error.to_string())?;
    drop(output);
    if assembling
        .metadata()
        .map_err(|error| error.to_string())?
        .len()
        != expected_size
    {
        return Err("assembled download has the wrong size".to_string());
    }
    if partial.exists() {
        fs::remove_file(partial).map_err(|error| error.to_string())?;
    }
    fs::rename(assembling, partial).map_err(|error| error.to_string())?;
    Ok(())
}

fn resolve_download_url(
    client: &reqwest::blocking::Client,
    url: &str,
    expected_size: u64,
) -> String {
    let response = match client.head(url).send() {
        Ok(response) if response.status().is_success() => response,
        Ok(response) => {
            log::warn!("download URL probe returned HTTP {}", response.status());
            return url.to_string();
        }
        Err(error) => {
            log::warn!("download URL probe failed: {error}");
            return url.to_string();
        }
    };
    if response
        .content_length()
        .is_some_and(|length| length != expected_size)
    {
        log::warn!("resolved download size does not match the release manifest");
        return url.to_string();
    }
    response.url().to_string()
}

fn download_file(
    app: &tauri::AppHandle,
    url: &str,
    dest: &Path,
    stage: &str,
    label: &str,
    expected_size: u64,
    expected_sha256: &str,
) -> Result<(), String> {
    let tmp = dest.with_extension("part");
    if file_matches(dest, expected_size, expected_sha256)? {
        if expected_size >= PARALLEL_DOWNLOAD_THRESHOLD {
            remove_download_chunks(&tmp);
        }
        let _ = fs::remove_file(&tmp);
        emit(
            app,
            "verify",
            &format!("{label} already downloaded and verified."),
            100,
        );
        return Ok(());
    }
    if dest.exists() {
        fs::remove_file(dest)
            .map_err(|e| format!("cannot remove invalid {}: {e}", dest.display()))?;
    }

    let client = reqwest::blocking::Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(DOWNLOAD_ATTEMPT_TIMEOUT)
        .http1_only()
        .tcp_nodelay(true)
        .user_agent(concat!("Mirid/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|e| e.to_string())?;
    let mut download_url = resolve_download_url(&client, url, expected_size);

    if expected_size >= PARALLEL_DOWNLOAD_THRESHOLD {
        let mut telemetry = DownloadTelemetry::new(0);
        let parallel_result = download_file_parallel(
            &client,
            &download_url,
            &tmp,
            expected_size,
            PARALLEL_DOWNLOAD_CONNECTIONS,
            PARALLEL_DOWNLOAD_SEGMENT_SIZE,
            |downloaded| {
                if let Some(snapshot) =
                    telemetry.observe(downloaded, expected_size, downloaded >= expected_size)
                {
                    emit_download(
                        app,
                        stage,
                        &format!(
                            "Downloading {label}  {:.1} / {:.1} GB",
                            downloaded as f64 / 1e9,
                            expected_size as f64 / 1e9
                        ),
                        snapshot,
                    );
                }
            },
        );
        match parallel_result {
            Ok(()) => {
                emit(app, "verify", &format!("Verifying {label}..."), 0);
                if file_matches(&tmp, expected_size, expected_sha256)? {
                    promote_download(&tmp, dest)?;
                    remove_download_chunks(&tmp);
                    emit(app, "verify", &format!("{label} verified."), 100);
                    return Ok(());
                }
                remove_download_chunks(&tmp);
                let _ = fs::remove_file(&tmp);
                return Err(format!("{label} failed SHA-256 verification"));
            }
            Err(error) => {
                log::warn!("parallel {label} download failed: {error}; using one connection");
                remove_download_chunks(&tmp);
                let _ = fs::remove_file(&tmp);
                emit(app, stage, &format!("Retrying {label}..."), 0);
                download_url = resolve_download_url(&client, url, expected_size);
            }
        }
    }
    let mut attempt = 0;

    loop {
        attempt += 1;

        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }

        let mut resume_from = if tmp.exists() {
            fs::metadata(&tmp).map_err(|e| e.to_string())?.len()
        } else {
            0
        };

        if resume_from == expected_size {
            emit(app, "verify", &format!("Verifying {label}..."), 0);
            if file_matches(&tmp, expected_size, expected_sha256)? {
                promote_download(&tmp, dest)?;
                emit(app, "verify", &format!("{label} verified."), 100);
                return Ok(());
            }
            fs::remove_file(&tmp).map_err(|e| e.to_string())?;
            resume_from = 0;
        } else if resume_from > expected_size {
            fs::remove_file(&tmp).map_err(|e| e.to_string())?;
            resume_from = 0;
        }

        let progress_message = if resume_from > 0 {
            format!("Resuming {label}...")
        } else {
            format!("Downloading {label}...")
        };
        let mut telemetry = DownloadTelemetry::new(resume_from);
        if let Some(snapshot) = telemetry.observe(resume_from, expected_size, true) {
            emit_download(app, stage, &progress_message, snapshot);
        }

        let mut req = client.get(&download_url);
        if resume_from > 0 {
            req = req.header("Range", format!("bytes={}-", resume_from));
        }

        let mut resp = match req.send() {
            Ok(r) => r,
            Err(e) => {
                log::warn!(
                    "{} download attempt {}/{} failed: {}",
                    label,
                    attempt,
                    MAX_RETRIES,
                    e
                );
                if attempt >= MAX_RETRIES {
                    return Err(format!(
                        "{} download failed after {} attempts: {}",
                        label, MAX_RETRIES, e
                    ));
                }
                let backoff = INITIAL_BACKOFF_MS * 2_u64.pow(attempt - 1);
                std::thread::sleep(Duration::from_millis(backoff));
                continue;
            }
        };

        if resp.status() == reqwest::StatusCode::RANGE_NOT_SATISFIABLE {
            let _ = fs::remove_file(&tmp);
            if attempt >= MAX_RETRIES {
                return Err(format!("{label} download range was rejected"));
            }
            continue;
        }

        if !resp.status().is_success() {
            let status = resp.status();
            if attempt >= MAX_RETRIES {
                return Err(format!("{} download HTTP {}", label, status));
            }
            log::warn!(
                "{} download HTTP {} (attempt {}/{}), retrying...",
                label,
                status,
                attempt,
                MAX_RETRIES
            );
            let backoff = INITIAL_BACKOFF_MS * 2_u64.pow(attempt - 1);
            std::thread::sleep(Duration::from_millis(backoff));
            continue;
        }

        if resume_from > 0 && resp.status() == reqwest::StatusCode::OK {
            log::warn!("{label} server ignored the Range header; restarting safely");
            let _ = fs::remove_file(&tmp);
            resume_from = 0;
        } else if resume_from > 0
            && (resp.status() != reqwest::StatusCode::PARTIAL_CONTENT
                || !content_range_starts_at(
                    resp.headers().get(reqwest::header::CONTENT_RANGE),
                    resume_from,
                ))
        {
            log::warn!("{label} returned an invalid Content-Range; restarting");
            let _ = fs::remove_file(&tmp);
            if attempt >= MAX_RETRIES {
                return Err(format!("{label} returned an invalid Content-Range"));
            }
            continue;
        }

        let expected_remaining = expected_size - resume_from;
        if let Some(response_size) = resp.content_length() {
            if response_size != expected_remaining {
                log::warn!(
                    "{label} response size changed ({} instead of {}), retrying",
                    response_size,
                    expected_remaining
                );
                let _ = fs::remove_file(&tmp);
                if attempt >= MAX_RETRIES {
                    return Err(format!(
                        "{label} response size does not match the release manifest"
                    ));
                }
                continue;
            }
        }

        let mut file = if resume_from > 0 {
            fs::OpenOptions::new()
                .write(true)
                .append(true)
                .open(&tmp)
                .map_err(|e| e.to_string())?
        } else {
            fs::File::create(&tmp).map_err(|e| e.to_string())?
        };

        let mut downloaded = resume_from;
        let mut buf = [0u8; 1024 * 256];
        let mut reached_eof = false;

        loop {
            let n = match resp.read(&mut buf) {
                Ok(n) => n,
                Err(e) => {
                    log::warn!(
                        "{} read error on attempt {}/{}: {}",
                        label,
                        attempt,
                        MAX_RETRIES,
                        e
                    );
                    break;
                }
            };
            if n == 0 {
                reached_eof = true;
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            downloaded += n as u64;

            if downloaded > expected_size {
                break;
            }

            if let Some(snapshot) =
                telemetry.observe(downloaded, expected_size, downloaded >= expected_size)
            {
                emit_download(
                    app,
                    stage,
                    &format!(
                        "Downloading {label}  {:.1} / {:.1} GB",
                        downloaded as f64 / 1e9,
                        expected_size as f64 / 1e9
                    ),
                    snapshot,
                );
            }
        }

        file.flush().map_err(|e| e.to_string())?;
        drop(file);

        if reached_eof && downloaded == expected_size {
            emit(app, "verify", &format!("Verifying {label}..."), 0);
            if file_matches(&tmp, expected_size, expected_sha256)? {
                promote_download(&tmp, dest)?;
                emit(app, "verify", &format!("{label} verified."), 100);
                return Ok(());
            }
            let _ = fs::remove_file(&tmp);
        }

        if attempt >= MAX_RETRIES {
            return Err(format!(
                "{} download incomplete after {} attempts ({} / {} bytes)",
                label, MAX_RETRIES, downloaded, expected_size
            ));
        }

        log::warn!(
            "{} download incomplete ({} / {} bytes), retrying...",
            label,
            downloaded,
            expected_size
        );
        let backoff = INITIAL_BACKOFF_MS * 2_u64.pow(attempt - 1);
        std::thread::sleep(Duration::from_millis(backoff));
        // loop continues, resuming from tmp file
    }
}

fn extract_runtime_archive(
    app: &tauri::AppHandle,
    archive: &Path,
    dest: &Path,
    previous_runtime: Option<&Path>,
) -> Result<(), String> {
    let archive_metadata = sevenz_rust::Archive::open(archive)
        .map_err(|error| format!("cannot read runtime archive: {error}"))?;
    let total_entries = archive_metadata.files.len();
    let total_bytes = archive_metadata
        .files
        .iter()
        .map(|entry| entry.size())
        .sum::<u64>();
    emit(
        app,
        "extract",
        &format!(
            "Preparing to install {} files ({})...",
            total_entries,
            format_runtime_size(total_bytes),
        ),
        0,
    );
    // Each process gets its own extraction and backup paths. Two launches can
    // then race safely: the first activates the runtime and the second reuses
    // the now-complete content-addressed destination.
    let staging = runtime_attempt_path(dest, "installing");
    let backup = runtime_attempt_path(dest, "previous");
    let _ = fs::remove_dir_all(&staging);
    let _ = fs::remove_dir_all(&backup);
    fs::create_dir_all(&staging).map_err(|e| e.to_string())?;

    let mut extracted_entries = 0usize;
    let mut extracted_bytes = 0u64;
    let mut last_percent = 0u8;
    let extraction_result = sevenz_rust::decompress_file_with_extract_fn(
        archive,
        &staging,
        |entry, reader, output_path| {
            if !archive_path_is_safe(entry.name()) {
                return Err(sevenz_rust::Error::other("archive contains an unsafe path"));
            }
            let result = {
                let mut progress_reader = ExtractionProgressReader {
                    inner: reader,
                    app,
                    extracted_bytes: &mut extracted_bytes,
                    total_bytes,
                    completed_entries: extracted_entries,
                    total_entries,
                    last_percent: &mut last_percent,
                };
                sevenz_rust::default_entry_extract_fn(entry, &mut progress_reader, output_path)
            };
            if result.is_ok() {
                extracted_entries += 1;
                if extracted_entries % 50 == 0 {
                    emit(
                        app,
                        "extract",
                        &format!(
                            "Installing runtime... {} of {} files · {} of {}",
                            extracted_entries,
                            total_entries,
                            format_runtime_size(extracted_bytes),
                            format_runtime_size(total_bytes),
                        ),
                        extraction_percent(extracted_bytes, total_bytes),
                    );
                }
            }
            result
        },
    );

    if let Err(error) = extraction_result {
        let _ = fs::remove_dir_all(&staging);
        return Err(format!("runtime extraction failed: {error}"));
    }
    if extracted_entries == 0 || !runtime_internal_is_complete(&staging) {
        let _ = fs::remove_dir_all(&staging);
        return Err("runtime archive did not contain a complete local engine".to_string());
    }

    emit(app, "extract", "Finishing the runtime installation...", 99);
    preserve_runtime_static_data(dest, &staging)?;
    if let Some(previous) = previous_runtime.filter(|previous| *previous != dest) {
        preserve_runtime_static_data(previous, &staging)?;
    }

    match activate_extracted_runtime(&staging, dest, &backup)? {
        RuntimeActivation::Activated => {
            let _ = fs::remove_dir_all(&backup);
        }
        RuntimeActivation::ReusedExisting => {
            log::info!(
                "The verified runtime already exists at {}; keeping it in place",
                dest.display()
            );
        }
    }
    emit(app, "extract", "Extraction complete.", 100);
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
enum RuntimeActivation {
    Activated,
    ReusedExisting,
}

fn activate_extracted_runtime(
    staging: &Path,
    dest: &Path,
    backup: &Path,
) -> Result<RuntimeActivation, String> {
    // The release directory is content-addressed by the runtime and sidecar
    // hashes. A complete destination is therefore the same immutable payload,
    // not an older version that needs replacing. Reuse it without renaming:
    // Windows may still have DLLs loaded from it after an uninstall.
    if runtime_internal_is_complete(dest) {
        let _ = fs::remove_dir_all(staging);
        return Ok(RuntimeActivation::ReusedExisting);
    }

    if backup.exists() {
        fs::remove_dir_all(backup)
            .map_err(|error| format!("cannot clear an interrupted runtime backup: {error}"))?;
    }
    if dest.exists() {
        fs::rename(dest, backup).map_err(|error| {
            format!(
                "cannot stage an incomplete previous runtime: {error}. Close any other copies of Mirid, then try again"
            )
        })?;
    }
    if let Err(error) = fs::rename(staging, dest) {
        if backup.exists() {
            let _ = fs::rename(backup, dest);
        }
        return Err(format!("cannot activate extracted runtime: {error}"));
    }
    Ok(RuntimeActivation::Activated)
}

fn ensure_runtime(app: &tauri::AppHandle) -> Result<(), String> {
    migrate_legacy_runtime(app)?;
    migrate_runtime_user_data(app)?;
    let dir = runtime_dir(app)?;
    if runtime_is_ready(app) {
        cleanup_download_artifacts(&dir);
        cleanup_runtime_staging_artifacts(&versioned_runtime_internal_dir(&dir));
        emit(app, "ready", "Runtime already installed.", 100);
        return Ok(());
    }

    fs::create_dir_all(&dir).map_err(|e| e.to_string())?;

    let internal = versioned_runtime_internal_dir(&dir);
    let exe_dest = versioned_sidecar_exe_path(&dir);
    if runtime_assets_are_reusable(&internal, &exe_dest, SIDECAR_EXE_SIZE, SIDECAR_EXE_SHA256)
        .unwrap_or(false)
    {
        cleanup_download_artifacts(&dir);
        cleanup_runtime_staging_artifacts(&internal);
        fs::write(ready_marker(app)?, RUNTIME_VERSION).map_err(|e| e.to_string())?;
        emit(
            app,
            "ready",
            "Existing runtime verified. Starting Mirid.",
            100,
        );
        return Ok(());
    }

    let previous_internal = existing_runtime_internal_dir(&dir);
    let _ = fs::remove_file(ready_marker(app)?);

    // 1) Sidecar exe.
    if let Some(parent) = exe_dest.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    let legacy_exe = legacy_sidecar_exe_path(&dir);
    if !exe_dest.exists()
        && file_matches(&legacy_exe, SIDECAR_EXE_SIZE, SIDECAR_EXE_SHA256).unwrap_or(false)
    {
        fs::copy(&legacy_exe, &exe_dest)
            .map_err(|error| format!("cannot stage the existing Mirid engine: {error}"))?;
    }
    download_file(
        app,
        &format!("{HF_BASE}/{SIDECAR_EXE}"),
        &exe_dest,
        "download",
        "Mirid",
        SIDECAR_EXE_SIZE,
        SIDECAR_EXE_SHA256,
    )?;
    make_executable(&exe_dest)?;

    // A reinstall may retain the immutable runtime while losing its ready
    // marker or sidecar. Once the exact sidecar has been restored, do not
    // download and replace the same 8.7 GB dependency tree.
    if runtime_internal_is_complete(&internal) {
        cleanup_download_artifacts(&dir);
        cleanup_runtime_staging_artifacts(&internal);
        fs::write(ready_marker(app)?, RUNTIME_VERSION).map_err(|e| e.to_string())?;
        emit(
            app,
            "ready",
            "Existing runtime repaired. Starting Mirid.",
            100,
        );
        return Ok(());
    }

    // 2) Runtime archive.
    let archive_dest = dir.join(RUNTIME_ARCHIVE);
    download_file(
        app,
        &format!("{HF_BASE}/{RUNTIME_ARCHIVE}"),
        &archive_dest,
        "download",
        "Mirid's local files",
        RUNTIME_ARCHIVE_SIZE,
        RUNTIME_ARCHIVE_SHA256,
    )?;

    // 3) Extract into an immutable, content-addressed runtime directory. A
    // previous running version can remain locked without blocking activation.
    extract_runtime_archive(app, &archive_dest, &internal, previous_internal.as_deref())?;
    let _ = fs::remove_file(&archive_dest);
    cleanup_download_artifacts(&dir);
    cleanup_runtime_staging_artifacts(&internal);

    // 4) Mark ready.
    fs::write(ready_marker(app)?, RUNTIME_VERSION).map_err(|e| e.to_string())?;
    emit(app, "ready", "Runtime installed.", 100);
    Ok(())
}

#[derive(Debug)]
struct DevelopmentVenv {
    project_root: PathBuf,
    python: PathBuf,
    entry_point: PathBuf,
    runner_root: PathBuf,
    runner_manifest: PathBuf,
}

fn development_venv_enabled() -> bool {
    cfg!(debug_assertions)
        && std::env::var("MIRID_DEV_USE_VENV")
            .map(|value| {
                matches!(
                    value.trim().to_ascii_lowercase().as_str(),
                    "1" | "true" | "yes" | "on"
                )
            })
            .unwrap_or(false)
}

fn development_venv() -> Result<Option<DevelopmentVenv>, String> {
    if !development_venv_enabled() {
        return Ok(None);
    }

    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(Path::to_path_buf)
        .ok_or_else(|| "Mirid's project root could not be resolved".to_string())?;
    let python = if cfg!(target_os = "windows") {
        project_root.join("venv").join("Scripts").join("python.exe")
    } else {
        project_root.join("venv").join("bin").join("python")
    };
    let entry_point = project_root.join("sidecar_entry.py");
    if !python.is_file() {
        return Err(format!(
            "Mirid's development virtual environment is missing: {}",
            python.display()
        ));
    }
    if !entry_point.is_file() {
        return Err(format!(
            "Mirid's development service entry point is missing: {}",
            entry_point.display()
        ));
    }

    let runner_root = project_root.join("build").join("model-runners");
    let staged_manifest = runner_root.join("manifest.json");
    let runner_manifest = if staged_manifest.is_file() {
        staged_manifest
    } else {
        project_root.join("runtime").join("model-runners.json")
    };

    Ok(Some(DevelopmentVenv {
        project_root,
        python,
        entry_point,
        runner_root,
        runner_manifest,
    }))
}

fn spawn_sidecar(
    app: &tauri::AppHandle,
    mode: &str,
    port: u16,
    tts_port: u16,
) -> Result<ManagedSidecar, String> {
    let development = development_venv()?;
    let (exe, working_directory, runners, runner_manifest, mut arguments, development_root) =
        if let Some(development) = development {
            log::info!(
                "Starting {mode} from Mirid's development venv: {}",
                development.python.display()
            );
            (
                development.python,
                Some(development.project_root.clone()),
                development.runner_root,
                development.runner_manifest,
                vec![development.entry_point.to_string_lossy().to_string()],
                Some(development.project_root),
            )
        } else {
            let runtime = runtime_dir(app)?;
            let layout = active_runtime_layout(&runtime)
                .ok_or_else(|| "Mirid's local engine is not installed correctly".to_string())?;
            let runners = layout.internal.join("runners");
            let runner_manifest = runners.join("manifest.json");
            (
                layout.sidecar,
                None,
                runners,
                runner_manifest,
                Vec::new(),
                None,
            )
        };
    let user_data = user_data_dir(app)?;
    let log_dir = user_data.join("logs");
    fs::create_dir_all(&user_data).map_err(|error| error.to_string())?;
    fs::create_dir_all(&log_dir).map_err(|error| error.to_string())?;
    let host = if mode == "backend" {
        backend_bind_host()
    } else {
        "127.0.0.1"
    };
    log::info!("Starting {mode} sidecar on {host}:{port}");
    arguments.extend([
        mode.to_string(),
        "--host".to_string(),
        host.to_string(),
        "--port".to_string(),
        port.to_string(),
    ]);
    let mut command = app
        .shell()
        .command(exe.to_string_lossy().to_string())
        .env("MIRID_DATA_DIR", user_data.to_string_lossy().to_string())
        .env("MIRID_LOG_DIR", log_dir.to_string_lossy().to_string())
        .env("ELOQUENT_LOG_DIR", log_dir.to_string_lossy().to_string())
        .env("MIRID_RUNNER_ROOT", runners.to_string_lossy().to_string())
        .env("PORT", port.to_string())
        .env("TTS_PORT", tts_port.to_string())
        .env("MIRID_SERVICE_ROLE", mode)
        .env(
            "MIRID_RUNNER_MANIFEST",
            runner_manifest.to_string_lossy().to_string(),
        );
    if let Some(working_directory) = working_directory {
        command = command.current_dir(working_directory);
    }
    if let Some(development_root) = development_root {
        command = command
            .env("PYTHONPATH", development_root.to_string_lossy().to_string())
            .env("PYTHONUTF8", "1")
            .env("PYTHONIOENCODING", "utf-8");
    }
    let command = command.args(arguments);

    let (mut events, child) = command.spawn().map_err(|error| error.to_string())?;
    let child = ManagedSidecar::new(child);
    let child_pid = child.pid();
    let child_exited = child.exit_flag();
    let mode = mode.to_owned();
    let handle = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = events.recv().await {
            match event {
                tauri_plugin_shell::process::CommandEvent::Stderr(bytes) => {
                    log::warn!("{mode}: {}", String::from_utf8_lossy(&bytes));
                }
                tauri_plugin_shell::process::CommandEvent::Terminated(payload) => {
                    child_exited.store(true, Ordering::Release);
                    log::info!("{mode} sidecar exited with {:?}", payload.code);
                    let sidecars = handle.state::<Sidecars>();
                    match mode.as_str() {
                        "backend" => {
                            if let Ok(mut child) = sidecars.backend.lock() {
                                if child.as_ref().map(ManagedSidecar::pid) == Some(child_pid) {
                                    *child = None;
                                }
                            }
                        }
                        "tts" => {
                            if let Ok(mut child) = sidecars.tts.lock() {
                                if child.as_ref().map(ManagedSidecar::pid) == Some(child_pid) {
                                    *child = None;
                                }
                            }
                        }
                        _ => {}
                    }
                    break;
                }
                _ => {}
            }
        }
    });

    Ok(child)
}

fn service_is_ready(client: &reqwest::blocking::Client, url: &str) -> bool {
    client
        .get(url)
        .send()
        .map(|response| response.status().is_success())
        .unwrap_or(false)
}

fn managed_service_endpoint_state(
    app: &tauri::AppHandle,
    label: &str,
    port: u16,
) -> Result<(bool, bool), String> {
    let sidecars = app.state::<Sidecars>();
    let child = match label {
        "backend" => sidecars
            .backend
            .lock()
            .map_err(|_| "backend sidecar lock poisoned")?,
        "voice service" => sidecars
            .tts
            .lock()
            .map_err(|_| "tts sidecar lock poisoned")?,
        _ => return Ok((true, true)),
    };
    let Some(child) = child.as_ref() else {
        return Ok((false, false));
    };
    if !child.is_running() {
        return Ok((false, false));
    }

    #[cfg(target_os = "windows")]
    {
        let listener_process_ids = windows_listener_process_ids(port)?;
        let owns_listener = listener_process_ids
            .into_iter()
            .any(|process_id| child.owns_process(process_id));
        Ok((true, owns_listener))
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = port;
        Ok((true, true))
    }
}

fn wait_for_service(
    app: &tauri::AppHandle,
    client: &reqwest::blocking::Client,
    label: &str,
    url: &str,
    message: &str,
    percent: u8,
) -> Result<(), String> {
    let port = reqwest::Url::parse(url)
        .map_err(|error| format!("invalid {label} health URL: {error}"))?
        .port_or_known_default()
        .ok_or_else(|| format!("{label} health URL has no port"))?;
    let started = Instant::now();
    let mut last_update = Instant::now();
    emit(app, "starting", message, percent);
    loop {
        let (sidecar_running, owns_listener) = managed_service_endpoint_state(app, label, port)?;
        if !sidecar_running {
            return Err(format!(
                "{label} process exited before its local endpoint became ready"
            ));
        }
        if owns_listener && service_is_ready(client, url) {
            return Ok(());
        }
        let elapsed = started.elapsed();
        if elapsed >= SERVICE_START_TIMEOUT {
            return Err(format!(
                "{label} did not become ready within {} seconds",
                SERVICE_START_TIMEOUT.as_secs()
            ));
        }
        if last_update.elapsed() >= Duration::from_secs(6) {
            emit(
                app,
                "starting",
                &format!("{message} {}s", elapsed.as_secs()),
                percent,
            );
            last_update = Instant::now();
        }
        std::thread::sleep(SERVICE_POLL_INTERVAL);
    }
}

fn stop_all_sidecars(app: &tauri::AppHandle) {
    let sidecars = app.state::<Sidecars>();
    if let Ok(mut backend) = sidecars.backend.lock() {
        if let Some(child) = backend.take() {
            let _ = child.kill();
        }
    }
    if let Ok(mut tts) = sidecars.tts.lock() {
        if let Some(child) = tts.take() {
            let _ = child.kill();
        }
    };
}

fn start_sidecars(app: &tauri::AppHandle) -> Result<(), String> {
    let runtime = app.state::<ServiceRuntime>();
    let _tts_operation = runtime
        .tts_operation
        .lock()
        .map_err(|_| "voice-service operation lock is unavailable".to_string())?;
    let sidecars = app.state::<Sidecars>();
    emit(app, "starting", "Starting local services.", 10);
    let (mut tts_port, tts_reservation) = runtime.take_tts_reservation()?;
    drop(tts_reservation);
    *sidecars
        .tts
        .lock()
        .map_err(|_| "tts sidecar lock poisoned")? =
        Some(spawn_sidecar(app, "tts", tts_port, tts_port)?);

    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(4))
        .build()
        .map_err(|error| error.to_string())?;
    let initial_voice_readiness = wait_for_service(
        app,
        &client,
        "voice service",
        &format!("http://127.0.0.1:{tts_port}/health"),
        "Starting voice services.",
        35,
    );
    if let Err(initial_error) = initial_voice_readiness {
        let tts_still_running = sidecars
            .tts
            .lock()
            .map_err(|_| "tts sidecar lock poisoned")?
            .as_ref()
            .map(ManagedSidecar::is_running)
            .unwrap_or(false);
        if tts_still_running {
            stop_all_sidecars(app);
            return Err(initial_error);
        }

        log::warn!(
            "Voice service exited before owning port {tts_port}; reserving the endpoint again and retrying once"
        );
        let retry_port = runtime.reserve_tts_for_restart(tts_port, Duration::ZERO)?;
        if retry_port != tts_port {
            log::warn!(
                "Voice port {tts_port} was claimed during startup; Mirid automatically selected 127.0.0.1:{retry_port}"
            );
            tts_port = retry_port;
        }
        let (reserved_tts_port, retry_reservation) = runtime.take_tts_reservation()?;
        debug_assert_eq!(tts_port, reserved_tts_port);
        drop(retry_reservation);
        *sidecars
            .tts
            .lock()
            .map_err(|_| "tts sidecar lock poisoned")? =
            Some(spawn_sidecar(app, "tts", tts_port, tts_port)?);
        publish_service_endpoints(app)?;
        if let Err(retry_error) = wait_for_service(
            app,
            &client,
            "voice service",
            &format!("http://127.0.0.1:{tts_port}/health"),
            "Retrying voice services on the reserved endpoint.",
            40,
        ) {
            stop_all_sidecars(app);
            return Err(format!(
                "{retry_error} (initial voice startup also failed: {initial_error})"
            ));
        }
    }

    let backend_reservation = runtime.take_backend_reservation()?;
    drop(backend_reservation);
    let backend = match spawn_sidecar(app, "backend", DEFAULT_BACKEND_PORT, tts_port) {
        Ok(child) => child,
        Err(error) => {
            stop_all_sidecars(app);
            return Err(error);
        }
    };
    *sidecars
        .backend
        .lock()
        .map_err(|_| "backend sidecar lock poisoned")? = Some(backend);
    if let Err(error) = wait_for_service(
        app,
        &client,
        "backend",
        &format!("http://127.0.0.1:{DEFAULT_BACKEND_PORT}/health"),
        "Starting the local engine. First launch can take a few minutes.",
        85,
    ) {
        stop_all_sidecars(app);
        return Err(error);
    }

    emit(app, "done", "Local services are ready.", 100);
    Ok(())
}

#[tauri::command]
fn sidecar_status(state: tauri::State<'_, Sidecars>) -> Result<SidecarStatus, String> {
    Ok(SidecarStatus {
        backend: state
            .backend
            .lock()
            .map_err(|_| "backend sidecar lock poisoned")?
            .as_ref()
            .map(ManagedSidecar::is_running)
            .unwrap_or(false),
        tts: state
            .tts
            .lock()
            .map_err(|_| "tts sidecar lock poisoned")?
            .as_ref()
            .map(ManagedSidecar::is_running)
            .unwrap_or(false),
    })
}

#[tauri::command]
fn get_service_endpoints(
    state: tauri::State<'_, ServiceRuntime>,
) -> Result<ServiceEndpoints, String> {
    state.endpoints()
}

#[tauri::command]
fn stop_tts(app: tauri::AppHandle) -> Result<(), String> {
    let runtime = app.state::<ServiceRuntime>();
    let _operation = runtime
        .tts_operation
        .lock()
        .map_err(|_| "voice-service operation lock is unavailable".to_string())?;
    let sidecars = app.state::<Sidecars>();
    if let Some(child) = sidecars
        .tts
        .lock()
        .map_err(|_| "tts sidecar lock poisoned")?
        .take()
    {
        child.kill().map_err(|error| error.to_string())?;
    }
    Ok(())
}

fn publish_service_endpoints(app: &tauri::AppHandle) -> Result<ServiceEndpoints, String> {
    let endpoints = app.state::<ServiceRuntime>().endpoints()?;
    if let Err(error) = app.emit("service-endpoints-changed", endpoints.clone()) {
        log::warn!("Could not publish updated service endpoints: {error}");
    }
    Ok(endpoints)
}

fn restart_tts_sync(app: &tauri::AppHandle) -> Result<ServiceEndpoints, String> {
    let runtime = app.state::<ServiceRuntime>();
    let _operation = runtime
        .tts_operation
        .lock()
        .map_err(|_| "voice-service operation lock is unavailable".to_string())?;
    let state = app.state::<Sidecars>();
    let previous_tts_port = runtime.current_tts_port()?;
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .timeout(Duration::from_secs(4))
        .build()
        .map_err(|error| error.to_string())?;
    let stopped_managed_tts = {
        let mut tts = state.tts.lock().map_err(|_| "tts sidecar lock poisoned")?;
        if let Some(child) = tts.take() {
            child.kill().map_err(|error| error.to_string())?;
            true
        } else {
            false
        }
    };

    let release_grace = if stopped_managed_tts {
        TTS_PORT_RELEASE_GRACE
    } else {
        Duration::ZERO
    };
    let tts_port = runtime.reserve_tts_for_restart(previous_tts_port, release_grace)?;
    let tts_port_changed = tts_port != previous_tts_port;
    if tts_port_changed {
        log::warn!(
            "Voice port {previous_tts_port} remained occupied during restart; Mirid reserved fallback endpoint 127.0.0.1:{tts_port}"
        );
    } else {
        log::info!("Reserved voice endpoint 127.0.0.1:{tts_port} for restart");
    }

    let mut backend_reservation = None;
    let restart_backend = if tts_port_changed {
        let backend = state
            .backend
            .lock()
            .map_err(|_| "backend sidecar lock poisoned")?
            .take();
        if let Some(child) = backend {
            child.kill().map_err(|error| error.to_string())?;
            backend_reservation = Some(reserve_fixed_service_port_after_release(
                backend_bind_host(),
                DEFAULT_BACKEND_PORT,
                TTS_PORT_RELEASE_GRACE,
            )?);
            log::warn!(
                "Restarting the main engine so its voice client follows fallback port {tts_port}"
            );
            true
        } else {
            false
        }
    } else {
        false
    };

    let (reserved_tts_port, tts_reservation) = runtime.take_tts_reservation()?;
    debug_assert_eq!(tts_port, reserved_tts_port);
    drop(tts_reservation);
    let child = spawn_sidecar(app, "tts", tts_port, tts_port)?;
    *state.tts.lock().map_err(|_| "tts sidecar lock poisoned")? = Some(child);
    if restart_backend {
        drop(backend_reservation.take());
        let backend = match spawn_sidecar(app, "backend", DEFAULT_BACKEND_PORT, tts_port) {
            Ok(child) => child,
            Err(error) => {
                stop_all_sidecars(app);
                return Err(format!(
                    "voice service selected port {tts_port}, but the main engine could not restart with the updated endpoint: {error}"
                ));
            }
        };
        *state
            .backend
            .lock()
            .map_err(|_| "backend sidecar lock poisoned")? = Some(backend);
    }

    if let Err(error) = wait_for_service(
        app,
        &client,
        "voice service",
        &format!("http://127.0.0.1:{tts_port}/health"),
        "Restarting voice services.",
        85,
    )
    .and_then(|_| {
        if restart_backend {
            wait_for_service(
                app,
                &client,
                "backend",
                &format!("http://127.0.0.1:{DEFAULT_BACKEND_PORT}/health"),
                "Restarting the local engine with the new voice endpoint.",
                90,
            )
        } else {
            Ok(())
        }
    }) {
        if restart_backend {
            stop_all_sidecars(app);
        } else if let Ok(mut tts) = state.tts.lock() {
            if let Some(child) = tts.take() {
                let _ = child.kill();
            }
        }
        return Err(error);
    }

    publish_service_endpoints(app)
}

#[tauri::command]
async fn restart_tts(app: tauri::AppHandle) -> Result<ServiceEndpoints, String> {
    tauri::async_runtime::spawn_blocking(move || restart_tts_sync(&app))
        .await
        .map_err(|error| error.to_string())?
}

#[tauri::command]
fn shutdown_app(app: tauri::AppHandle) -> Result<(), String> {
    stop_all_sidecars(&app);
    app.exit(0);
    Ok(())
}

#[tauri::command]
fn restart_app(app: tauri::AppHandle) -> Result<(), String> {
    stop_all_sidecars(&app);
    app.restart()
}

const INSTALLER_AUDIO_PROFILE: &str = "installer-audio.ini";
const MAX_INSTALLER_AUDIO_PROFILE_SIZE: u64 = 64 * 1024;

#[derive(Debug, Default, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
struct InstallerAudioProfile {
    #[serde(skip_serializing_if = "Option::is_none")]
    tts_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stt_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tts_engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stt_engine: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nanogpt_stt_model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    nano_gpt_api_key: Option<String>,
}

fn parse_installer_bool(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        _ => None,
    }
}

fn parse_installer_audio_profile(contents: &str) -> Option<InstallerAudioProfile> {
    const STT_ENGINES: &[&str] = &[
        "whisper",
        "whisper3",
        "parakeet",
        "parakeet-v3",
        "parakeet-zh",
        "nemotron",
        "moonshine",
        "parakeet-cpp",
        "nanogpt",
    ];
    const TTS_ENGINES: &[&str] = &[
        "kokoro",
        "chatterbox",
        "chatterbox_turbo",
        "chatterbox_nano",
        "voxcpm",
        "voxcpm-gguf",
        "nanogpt-Qwen-3-TTS-1.7B",
    ];

    let mut profile = InstallerAudioProfile::default();
    let mut in_audio_section = false;
    for raw_line in contents.lines() {
        let line = raw_line.trim().trim_start_matches('\u{feff}');
        if line.is_empty() || line.starts_with(';') || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            in_audio_section = line[1..line.len() - 1].trim().eq_ignore_ascii_case("audio");
            continue;
        }
        if !in_audio_section {
            continue;
        }
        let Some((raw_key, raw_value)) = line.split_once('=') else {
            continue;
        };
        let key = raw_key.trim().to_ascii_lowercase();
        let value = raw_value.trim();
        match key.as_str() {
            "ttsenabled" => profile.tts_enabled = parse_installer_bool(value),
            "sttenabled" => profile.stt_enabled = parse_installer_bool(value),
            "ttsengine" if TTS_ENGINES.contains(&value) => {
                profile.tts_engine = Some(value.to_string())
            }
            "sttengine" if STT_ENGINES.contains(&value) => {
                profile.stt_engine = Some(value.to_string())
            }
            "nanogptsttmodel" if !value.is_empty() && value.len() <= 256 => {
                profile.nanogpt_stt_model = Some(value.to_string())
            }
            "nanogptapikey" if !value.is_empty() && value.len() <= 4096 => {
                profile.nano_gpt_api_key = Some(value.to_string())
            }
            _ => {}
        }
    }

    if profile == InstallerAudioProfile::default() {
        None
    } else {
        Some(profile)
    }
}

fn installer_audio_profile_path(app: &tauri::AppHandle) -> Result<PathBuf, String> {
    app.path()
        .app_local_data_dir()
        .map(|path| path.join(INSTALLER_AUDIO_PROFILE))
        .map_err(|error| error.to_string())
}

#[tauri::command]
fn read_installer_audio_profile(
    app: tauri::AppHandle,
) -> Result<Option<InstallerAudioProfile>, String> {
    let path = installer_audio_profile_path(&app)?;
    if !path.is_file() {
        return Ok(None);
    }
    let metadata = fs::metadata(&path).map_err(|error| error.to_string())?;
    if metadata.len() > MAX_INSTALLER_AUDIO_PROFILE_SIZE {
        return Err("installer audio profile is unexpectedly large".to_string());
    }
    let contents = fs::read_to_string(path).map_err(|error| error.to_string())?;
    Ok(parse_installer_audio_profile(&contents))
}

#[tauri::command]
fn clear_installer_audio_profile(app: tauri::AppHandle) -> Result<(), String> {
    let path = installer_audio_profile_path(&app)?;
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.to_string()),
    }
}

#[tauri::command]
fn get_app_info(
    app: tauri::AppHandle,
    services: tauri::State<'_, ServiceRuntime>,
) -> Result<serde_json::Value, String> {
    let tts_port = services.current_tts_port()?;
    let log_dir = app
        .path()
        .app_log_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();
    let runtime_path = runtime_dir(&app).ok();
    let runtime_download_size = runtime_path
        .as_deref()
        .map(versioned_runtime_internal_dir)
        .filter(|internal| runtime_internal_is_complete(internal))
        .map(|_| SIDECAR_EXE_SIZE)
        .unwrap_or(RUNTIME_ARCHIVE_SIZE + SIDECAR_EXE_SIZE);
    let runtime_dir = runtime_path
        .map(|path| path.to_string_lossy().to_string())
        .unwrap_or_default();
    Ok(serde_json::json!({
      "log_dir": log_dir,
      "runtime_dir": runtime_dir,
      "runtime_version": RUNTIME_VERSION,
      "runtime_download_size": runtime_download_size,
      "runtime_installed_size": RUNTIME_INSTALLED_SIZE,
      "model_runner_contract_version": MODEL_RUNNER_CONTRACT_VERSION,
      "runtime_ready": runtime_is_ready(&app),
      "backend_port": DEFAULT_BACKEND_PORT,
      "tts_port": tts_port,
    }))
}

#[tauri::command]
fn open_runtime_folder(app: tauri::AppHandle) -> Result<(), String> {
    let directory = runtime_dir(&app)?;
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    #[cfg(target_os = "windows")]
    let mut command = std::process::Command::new("explorer.exe");
    #[cfg(target_os = "macos")]
    let mut command = std::process::Command::new("open");
    command
        .arg(&directory)
        .spawn()
        .map_err(|error| format!("cannot open runtime folder: {error}"))?;
    Ok(())
}

#[tauri::command]
fn read_log_tail(app: tauri::AppHandle, lines: usize) -> Result<String, String> {
    use std::io::BufRead;
    let current_log_dir = app
        .path()
        .app_log_dir()
        .map_err(|error| error.to_string())?;
    #[cfg(target_os = "windows")]
    let app_data_root = current_log_dir
        .parent()
        .and_then(Path::parent)
        .unwrap_or_else(|| Path::new(""));
    #[cfg(target_os = "windows")]
    let legacy_log_dir = app_data_root.join(LEGACY_APP_ID).join("logs");
    #[cfg(not(target_os = "windows"))]
    let legacy_log_dir = current_log_dir.clone();
    let log_dir = if current_log_dir.exists() {
        current_log_dir
    } else {
        legacy_log_dir
    };
    let mut latest: Option<(std::time::SystemTime, PathBuf)> = None;
    if let Ok(entries) = fs::read_dir(&log_dir) {
        for e in entries.flatten() {
            if let Some(ext) = e.path().extension() {
                if ext == "log" {
                    if let Ok(meta) = e.metadata() {
                        if let Ok(modified) = meta.modified() {
                            if latest.as_ref().map(|(t, _)| modified > *t).unwrap_or(true) {
                                latest = Some((modified, e.path()));
                            }
                        }
                    }
                }
            }
        }
    }
    let path = match latest {
        Some((_, p)) => p,
        None => return Err("no log file found".to_string()),
    };
    let file = fs::File::open(&path).map_err(|e| e.to_string())?;
    let reader = BufReader::new(file);
    let all: Vec<String> = reader.lines().map_while(Result::ok).collect();
    let start = all.len().saturating_sub(lines.max(1));
    Ok(all[start..].join("\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderValue;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_file(contents: &[u8]) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("mirid-runtime-test-{suffix}"));
        fs::write(&path, contents).expect("temporary file should be writable");
        path
    }

    fn temporary_directory(label: &str) -> PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after the Unix epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!("mirid-{label}-{suffix}"));
        fs::create_dir_all(&path).expect("temporary directory should be writable");
        path
    }

    #[test]
    fn only_the_main_window_owns_sidecar_shutdown() {
        assert!(destroyed_window_owns_sidecars("main"));
        assert!(!destroyed_window_owns_sidecars("settings"));
        assert!(!destroyed_window_owns_sidecars("secondary"));
    }

    #[test]
    fn reserves_the_preferred_service_port_when_it_is_available() {
        let candidate = TcpListener::bind(("127.0.0.1", 0))
            .expect("test should obtain an available loopback port")
            .local_addr()
            .expect("test listener should have an address")
            .port();

        let (selected, reservation) = reserve_service_port("127.0.0.1", candidate, "test service")
            .expect("available preferred port should be reserved");

        assert_eq!(selected, candidate);
        assert_eq!(
            reservation
                .local_addr()
                .expect("reservation should have an address")
                .port(),
            candidate
        );
    }

    #[test]
    fn automatically_reserves_another_port_when_the_preferred_port_is_occupied() {
        let occupied =
            TcpListener::bind(("127.0.0.1", 0)).expect("test should occupy a loopback port");
        let occupied_port = occupied
            .local_addr()
            .expect("occupied listener should have an address")
            .port();

        let (selected, reservation) =
            reserve_service_port("127.0.0.1", occupied_port, "test service")
                .expect("occupied preferred port should use an automatic fallback");

        assert_ne!(selected, occupied_port);
        assert_eq!(
            reservation
                .local_addr()
                .expect("fallback reservation should have an address")
                .port(),
            selected
        );
    }

    #[test]
    fn fixed_backend_port_reports_a_clear_conflict_instead_of_falling_back() {
        let occupied =
            TcpListener::bind(("127.0.0.1", 0)).expect("test should occupy a loopback port");
        let occupied_port = occupied
            .local_addr()
            .expect("occupied listener should have an address")
            .port();

        let error = match reserve_fixed_service_port("127.0.0.1", occupied_port) {
            Ok(_) => panic!("fixed backend reservation should not select another port"),
            Err(error) => error,
        };

        assert!(error.contains(&format!("Main engine port {occupied_port}")));
        assert!(error.contains("Close the other program or Mirid session"));
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn windows_listener_table_identifies_the_process_that_owns_a_port() {
        let listener =
            TcpListener::bind(("127.0.0.1", 0)).expect("test should bind a loopback listener");
        let port = listener
            .local_addr()
            .expect("test listener should have an address")
            .port();

        let process_ids =
            windows_listener_process_ids(port).expect("Windows listener table should be readable");

        assert!(
            process_ids.contains(&std::process::id()),
            "current process should own the test listener"
        );
    }

    #[test]
    fn service_endpoints_expose_the_ports_selected_by_the_desktop_host() {
        let runtime = ServiceRuntime {
            backend_reservation: Mutex::new(None),
            tts: Mutex::new(TtsRuntime {
                port: 18102,
                reservation: None,
            }),
            tts_operation: Mutex::new(()),
        };

        let endpoints = runtime
            .endpoints()
            .expect("service endpoint state should be readable");

        assert_eq!(endpoints.backend, "http://127.0.0.1:8000");
        assert_eq!(endpoints.secondary, endpoints.backend);
        assert_eq!(endpoints.tts, "http://127.0.0.1:18102");
        assert_eq!(endpoints.backend_port, 8000);
        assert_eq!(endpoints.tts_port, 18102);
    }

    #[test]
    fn tts_restart_fallback_updates_shared_port_state_and_holds_the_reservation() {
        let occupied =
            TcpListener::bind(("127.0.0.1", 0)).expect("test should occupy a loopback port");
        let occupied_port = occupied
            .local_addr()
            .expect("occupied listener should have an address")
            .port();
        let runtime = ServiceRuntime {
            backend_reservation: Mutex::new(None),
            tts: Mutex::new(TtsRuntime {
                port: occupied_port,
                reservation: None,
            }),
            tts_operation: Mutex::new(()),
        };

        let selected = runtime
            .reserve_tts_for_restart(occupied_port, Duration::ZERO)
            .expect("restart should reserve a fallback voice port");
        let (state_port, reservation) = runtime
            .take_tts_reservation()
            .expect("selected fallback should remain reserved until spawn");

        assert_ne!(selected, occupied_port);
        assert_eq!(state_port, selected);
        assert_eq!(
            reservation
                .local_addr()
                .expect("reservation should have an address")
                .port(),
            selected
        );
    }

    #[test]
    fn parses_installer_audio_choices() {
        let profile = parse_installer_audio_profile(
            "[audio]\nttsEnabled=1\nsttEnabled=true\nttsEngine=nanogpt-Qwen-3-TTS-1.7B\nsttEngine=nanogpt\nnanogptSttModel=fun-asr-flash-2026-06-15\nnanoGptApiKey=test-key\n",
        )
        .expect("valid installer profile should parse");

        assert_eq!(profile.tts_enabled, Some(true));
        assert_eq!(profile.stt_enabled, Some(true));
        assert_eq!(
            profile.tts_engine.as_deref(),
            Some("nanogpt-Qwen-3-TTS-1.7B")
        );
        assert_eq!(profile.stt_engine.as_deref(), Some("nanogpt"));
        assert_eq!(
            profile.nanogpt_stt_model.as_deref(),
            Some("fun-asr-flash-2026-06-15")
        );
        assert_eq!(profile.nano_gpt_api_key.as_deref(), Some("test-key"));
    }

    #[test]
    fn ignores_unsupported_installer_audio_engines() {
        let profile = parse_installer_audio_profile(
            "[audio]\nttsEnabled=0\nttsEngine=unknown\nsttEngine=unknown\n",
        )
        .expect("the valid enable flag should remain");

        assert_eq!(profile.tts_enabled, Some(false));
        assert_eq!(profile.tts_engine, None);
        assert_eq!(profile.stt_engine, None);
    }

    fn create_complete_runtime_internal(path: &Path) {
        fs::create_dir_all(path.join("backend"))
            .expect("runtime backend directory should be writable");
        #[cfg(target_os = "windows")]
        fs::write(path.join("python312.dll"), b"python")
            .expect("runtime Python DLL should be writable");
    }

    #[test]
    fn copies_runtime_user_data_without_overwriting_newer_files() {
        let source = temporary_directory("runtime-source");
        let destination = temporary_directory("runtime-destination");
        fs::create_dir_all(source.join("nested")).expect("source directory should be writable");
        fs::write(source.join("nested").join("history.json"), b"old")
            .expect("source file should be writable");
        fs::create_dir_all(destination.join("nested"))
            .expect("destination directory should be writable");
        fs::write(destination.join("nested").join("history.json"), b"new")
            .expect("destination file should be writable");

        copy_dir_contents(&source, &destination, false).expect("migration should succeed");

        assert_eq!(
            fs::read(destination.join("nested").join("history.json"))
                .expect("destination file should remain readable"),
            b"new"
        );
        let _ = fs::remove_dir_all(source);
        let _ = fs::remove_dir_all(destination);
    }

    #[test]
    fn preserves_generated_media_and_room_gallery_across_runtime_updates() {
        let current = temporary_directory("static-media-current");
        let staging = temporary_directory("static-media-staging");
        let generated = Path::new("backend/app/static/generated_images/example.png");
        let gallery_image = Path::new("backend/app/static/room_gallery/saved.png");
        let gallery_manifest = Path::new("backend/app/static/room_gallery/gallery_manifest.json");

        for relative in [generated, gallery_image, gallery_manifest] {
            let source = current.join(relative);
            fs::create_dir_all(source.parent().expect("fixture should have a parent"))
                .expect("fixture directory should be writable");
            fs::write(&source, relative.to_string_lossy().as_bytes())
                .expect("fixture should be writable");
        }

        preserve_runtime_static_data(&current, &staging)
            .expect("runtime media preservation should succeed");

        for relative in [generated, gallery_image, gallery_manifest] {
            assert_eq!(
                fs::read(staging.join(relative)).expect("preserved file should be readable"),
                relative.to_string_lossy().as_bytes()
            );
        }

        let _ = fs::remove_dir_all(current);
        let _ = fs::remove_dir_all(staging);
    }

    #[test]
    fn migrates_uploaded_avatars_but_not_packaged_static_assets() {
        let source = temporary_directory("avatar-source");
        let destination = temporary_directory("avatar-destination");
        let avatar = "0f9e8d7c-6b5a-4321-9fed-cba987654321.png";
        fs::write(source.join(avatar), b"portrait").expect("avatar should be writable");
        fs::write(source.join("packaged-logo.png"), b"logo")
            .expect("packaged asset should be writable");
        fs::create_dir_all(source.join("generated_images"))
            .expect("generated image directory should be writable");
        fs::write(
            source
                .join("generated_images")
                .join("11111111-2222-3333-4444-555555555555.png"),
            b"generated",
        )
        .expect("generated image should be writable");
        fs::create_dir_all(&destination).expect("destination should be writable");
        fs::write(destination.join(avatar), b"newer portrait")
            .expect("existing avatar should be writable");

        copy_runtime_avatar_files(&source, &destination).expect("avatar migration should succeed");

        assert_eq!(
            fs::read(destination.join(avatar)).expect("avatar should remain readable"),
            b"newer portrait"
        );
        assert!(!destination.join("packaged-logo.png").exists());
        assert!(!destination.join("generated_images").exists());
        let _ = fs::remove_dir_all(source);
        let _ = fs::remove_dir_all(destination);
    }

    #[test]
    fn finds_prior_versioned_runtime_for_user_data_migration() {
        let runtime = temporary_directory("prior-runtime-candidate");
        let prior_internal = runtime
            .join("releases")
            .join("v8-prior-archive-prior-sidecar")
            .join("_internal");
        create_complete_runtime_internal(&prior_internal);

        assert!(installed_runtime_internal_dirs(&runtime)
            .iter()
            .any(|candidate| candidate == &prior_internal));
        let _ = fs::remove_dir_all(runtime);
    }

    #[test]
    fn migrates_newest_runtime_avatar_without_overwriting_persistent_data() {
        let runtime = temporary_directory("avatar-runtime-migration");
        let destination = temporary_directory("avatar-data-migration");
        let avatar = "0f9e8d7c-6b5a-4321-9fed-cba987654321.png";
        let older_internal = runtime.join("releases").join("v8-old").join("_internal");
        let newer_internal = runtime.join("releases").join("v10-new").join("_internal");
        create_complete_runtime_internal(&older_internal);
        create_complete_runtime_internal(&newer_internal);
        let older_static = older_internal.join("backend").join("app").join("static");
        let newer_static = newer_internal.join("backend").join("app").join("static");
        fs::create_dir_all(&older_static).expect("older static directory should be writable");
        fs::create_dir_all(&newer_static).expect("newer static directory should be writable");
        fs::write(older_static.join(avatar), b"older").expect("older avatar should be writable");
        fs::write(newer_static.join(avatar), b"newer").expect("newer avatar should be writable");
        let older_data = older_internal.join("backend").join("app").join("data");
        fs::create_dir_all(&older_data).expect("older data directory should be writable");
        fs::write(older_data.join("stale.json"), b"stale").expect("older data should be writable");

        migrate_runtime_user_data_from_paths(&runtime, &destination)
            .expect("runtime migration should succeed");
        assert_eq!(
            fs::read(destination.join("avatars").join(avatar))
                .expect("migrated avatar should be readable"),
            b"newer"
        );
        assert!(
            !destination.join("stale.json").exists(),
            "avatar recovery must not restore unrelated data from older runtimes"
        );

        fs::write(destination.join("avatars").join(avatar), b"persistent")
            .expect("persistent avatar should be writable");
        migrate_runtime_user_data_from_paths(&runtime, &destination)
            .expect("repeated migration should succeed");
        assert_eq!(
            fs::read(destination.join("avatars").join(avatar))
                .expect("persistent avatar should remain readable"),
            b"persistent"
        );
        let _ = fs::remove_dir_all(runtime);
        let _ = fs::remove_dir_all(destination);
    }

    #[test]
    fn keeps_runtime_releases_in_side_by_side_paths() {
        let runtime = temporary_directory("versioned-runtime");
        let release = versioned_runtime_release_dir(&runtime);
        let internal = versioned_runtime_internal_dir(&runtime);
        let sidecar = versioned_sidecar_exe_path(&runtime);

        assert_ne!(internal, legacy_runtime_internal_dir(&runtime));
        assert_ne!(sidecar, legacy_sidecar_exe_path(&runtime));
        assert_eq!(internal.parent(), Some(release.as_path()));
        assert_eq!(sidecar.parent(), Some(release.as_path()));
        assert_eq!(
            internal.file_name().and_then(|name| name.to_str()),
            Some("_internal")
        );
        assert_eq!(
            sidecar.extension().and_then(|value| value.to_str()),
            Some("exe")
        );
        let _ = fs::remove_dir_all(runtime);
    }

    #[test]
    fn reuses_complete_content_addressed_runtime_assets() {
        let runtime = temporary_directory("reusable-runtime");
        let internal = versioned_runtime_internal_dir(&runtime);
        let sidecar = versioned_sidecar_exe_path(&runtime);
        create_complete_runtime_internal(&internal);
        fs::write(&sidecar, b"abc").expect("sidecar should be writable");
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";

        assert!(
            runtime_assets_are_reusable(&internal, &sidecar, 3, expected)
                .expect("runtime verification should succeed")
        );

        fs::remove_dir_all(internal.join("backend"))
            .expect("test should make the runtime incomplete");
        assert!(
            !runtime_assets_are_reusable(&internal, &sidecar, 3, expected)
                .expect("incomplete runtime should be rejected")
        );
        let _ = fs::remove_dir_all(runtime);
    }

    #[test]
    fn activation_keeps_a_complete_existing_runtime_in_place() {
        let runtime = temporary_directory("runtime-reinstall");
        let dest = runtime.join("_internal");
        let staging = runtime.join("_internal.installing");
        let backup = runtime.join("_internal.previous");
        create_complete_runtime_internal(&dest);
        create_complete_runtime_internal(&staging);
        fs::write(dest.join("existing.txt"), b"keep")
            .expect("existing runtime marker should be writable");
        fs::write(staging.join("fresh.txt"), b"replace")
            .expect("staged runtime marker should be writable");

        assert_eq!(
            activate_extracted_runtime(&staging, &dest, &backup)
                .expect("complete runtime should be reusable"),
            RuntimeActivation::ReusedExisting
        );
        assert_eq!(
            fs::read(dest.join("existing.txt")).expect("existing runtime should remain readable"),
            b"keep"
        );
        assert!(!dest.join("fresh.txt").exists());
        assert!(!staging.exists());
        assert!(!backup.exists());
        let _ = fs::remove_dir_all(runtime);
    }

    #[test]
    fn removes_abandoned_runtime_staging_without_touching_runtime_or_backup() {
        let runtime = temporary_directory("runtime-staging-cleanup");
        let internal = runtime.join("_internal");
        let legacy_staging = runtime.join("_internal.installing");
        let attempt_staging = runtime.join("_internal.installing-123-456");
        let backup = runtime.join("_internal.previous");
        for path in [&internal, &legacy_staging, &attempt_staging, &backup] {
            fs::create_dir_all(path).expect("fixture directory should be writable");
        }

        cleanup_runtime_staging_artifacts(&internal);

        assert!(internal.is_dir());
        assert!(!legacy_staging.exists());
        assert!(!attempt_staging.exists());
        assert!(backup.is_dir());
        let _ = fs::remove_dir_all(runtime);
    }

    #[test]
    fn accepts_legacy_runtime_assets_until_the_next_runtime_release() {
        let runtime = temporary_directory("legacy-runtime");
        let legacy_internal = legacy_runtime_internal_dir(&runtime);
        let legacy_sidecar = legacy_sidecar_exe_path(&runtime);
        create_complete_runtime_internal(&legacy_internal);
        fs::write(&legacy_sidecar, b"sidecar").expect("legacy sidecar should be writable");

        assert_eq!(
            active_runtime_layout(&runtime),
            Some(RuntimeLayout {
                internal: legacy_internal,
                sidecar: legacy_sidecar,
            })
        );
        let _ = fs::remove_dir_all(runtime);
    }

    #[test]
    fn rejects_a_sidecar_paired_with_another_release_dependencies() {
        let runtime = temporary_directory("sidecar-only-runtime");
        let prior_release = runtime.join("releases").join(format!(
            "prior-runtime-{}-prior-sidecar",
            &RUNTIME_ARCHIVE_SHA256[..12]
        ));
        let prior_internal = prior_release.join("_internal");
        let current_sidecar = versioned_sidecar_exe_path(&runtime);
        create_complete_runtime_internal(&prior_internal);
        fs::create_dir_all(
            current_sidecar
                .parent()
                .expect("sidecar should have a parent"),
        )
        .expect("current release directory should be writable");
        fs::write(&current_sidecar, b"new sidecar").expect("current sidecar should be writable");

        assert_eq!(active_runtime_layout(&runtime), None);
        let _ = fs::remove_dir_all(runtime);
    }

    #[test]
    fn rejects_installed_dependencies_from_another_archive() {
        let runtime = temporary_directory("different-archive-runtime");
        let incompatible_internal = runtime
            .join("releases")
            .join(format!("{RUNTIME_VERSION}-000000000000-prior-sidecar"))
            .join("_internal");
        let current_sidecar = versioned_sidecar_exe_path(&runtime);
        create_complete_runtime_internal(&incompatible_internal);
        fs::create_dir_all(
            current_sidecar
                .parent()
                .expect("sidecar should have a parent"),
        )
        .expect("current release directory should be writable");
        fs::write(current_sidecar, b"new sidecar").expect("current sidecar should be writable");

        assert_eq!(active_runtime_layout(&runtime), None);
        let _ = fs::remove_dir_all(runtime);
    }

    #[test]
    fn never_combines_assets_from_different_runtime_layouts() {
        let runtime = temporary_directory("mixed-runtime");
        create_complete_runtime_internal(&versioned_runtime_internal_dir(&runtime));
        fs::write(legacy_sidecar_exe_path(&runtime), b"legacy sidecar")
            .expect("legacy sidecar should be writable");

        assert_eq!(active_runtime_layout(&runtime), None);
        let _ = fs::remove_dir_all(runtime);
    }

    #[test]
    fn hashes_files_with_sha256() {
        let path = temporary_file(b"abc");
        let hash = sha256_file(&path).expect("hashing should succeed");
        let _ = fs::remove_file(path);
        assert_eq!(
            hash,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn verifies_file_size_and_hash() {
        let path = temporary_file(b"abc");
        let expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        assert!(file_matches(&path, 3, expected).expect("verification should succeed"));
        assert!(!file_matches(&path, 4, expected).expect("size mismatch should be handled"));
        assert!(!file_matches(&path, 3, &"0".repeat(64)).expect("hash mismatch should be handled"));
        let _ = fs::remove_file(path);
    }

    #[test]
    fn keeps_backend_local_without_lan_password() {
        assert_eq!(
            backend_host_from_settings(&serde_json::json!({})),
            "127.0.0.1"
        );
        assert_eq!(
            backend_host_from_settings(&serde_json::json!({
                "openaiServerLanEnabled": true,
                "admin_password": ""
            })),
            "127.0.0.1"
        );
    }

    #[test]
    fn exposes_backend_to_lan_only_with_password() {
        assert_eq!(
            backend_host_from_settings(&serde_json::json!({
                "openaiServerLanEnabled": true,
                "admin_password": "secret"
            })),
            "0.0.0.0"
        );
    }

    #[test]
    fn validates_content_range_start() {
        let valid = HeaderValue::from_static("bytes 1024-2047/4096");
        let invalid = HeaderValue::from_static("bytes 0-2047/4096");
        assert!(content_range_starts_at(Some(&valid), 1024));
        assert!(!content_range_starts_at(Some(&invalid), 1024));
        assert!(!content_range_starts_at(None, 1024));
    }

    #[test]
    fn rejects_unsafe_archive_paths() {
        assert!(archive_path_is_safe("backend/models/model.bin"));
        assert!(!archive_path_is_safe("../outside.txt"));
        assert!(!archive_path_is_safe("/absolute.txt"));
        assert!(!archive_path_is_safe("C:\\absolute.txt"));
    }

    #[test]
    fn reports_extraction_progress_without_claiming_completion_early() {
        assert_eq!(extraction_percent(0, 100), 0);
        assert_eq!(extraction_percent(50, 100), 50);
        assert_eq!(extraction_percent(100, 100), 99);
        assert_eq!(extraction_percent(100, 0), 0);
    }

    #[test]
    fn splits_large_downloads_into_small_balanced_segments() {
        assert_eq!(download_ranges(10, 4), vec![(0, 3), (4, 7), (8, 9)]);
        assert!(download_ranges(0, 4).is_empty());
    }

    #[test]
    fn reports_download_bytes_before_a_rate_sample_exists() {
        let mut telemetry = DownloadTelemetry::new(0);
        assert_eq!(
            telemetry.observe(25, 100, true),
            Some(DownloadSnapshot {
                downloaded_bytes: 25,
                total_bytes: 100,
                bytes_per_second: None,
                eta_seconds: None,
                percent: 25,
            })
        );
    }

    #[test]
    fn accepts_successful_service_health_response() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("test server should bind");
        let address = listener
            .local_addr()
            .expect("test server should have an address");
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("health request should connect");
            let mut request = [0u8; 1024];
            let _ = stream.read(&mut request);
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\n{}")
                .expect("health response should be written");
        });
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .expect("test client should build");
        assert!(service_is_ready(
            &client,
            &format!("http://{address}/health")
        ));
        server.join().expect("test server should exit");
    }

    #[test]
    fn downloads_file_across_parallel_ranges() {
        let contents = (0..2 * 1024 * 1024)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let server_contents = std::sync::Arc::new(contents.clone());
        let listener = TcpListener::bind("127.0.0.1:0").expect("test server should bind");
        let address = listener
            .local_addr()
            .expect("test server should have an address");
        let server = std::thread::spawn(move || {
            for _ in 0..8 {
                let (mut stream, _) = listener.accept().expect("range request should connect");
                let mut request = [0u8; 4096];
                let length = stream
                    .read(&mut request)
                    .expect("request should be readable");
                let request = String::from_utf8_lossy(&request[..length]);
                let range = request
                    .lines()
                    .find(|line| line.to_ascii_lowercase().starts_with("range: bytes="))
                    .expect("request should include a byte range")
                    .split_once(':')
                    .expect("range header should have a value")
                    .1
                    .trim()
                    .strip_prefix("bytes=")
                    .expect("range value should use bytes")
                    .split_once('-')
                    .expect("range should have start and end");
                let start = range.0.parse::<usize>().expect("range start should parse");
                let end = range.1.parse::<usize>().expect("range end should parse");
                let body = &server_contents[start..=end];
                let headers = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\nContent-Range: bytes {}-{}/{}\r\nConnection: close\r\n\r\n",
                    body.len(),
                    start,
                    end,
                    server_contents.len()
                );
                stream
                    .write_all(headers.as_bytes())
                    .expect("range headers should be written");
                stream
                    .write_all(body)
                    .expect("range body should be written");
            }
        });

        let partial = temporary_file(&[]);
        let client = reqwest::blocking::Client::builder()
            .http1_only()
            .build()
            .expect("test client should build");
        download_file_parallel(
            &client,
            &format!("http://{address}/runtime"),
            &partial,
            contents.len() as u64,
            4,
            256 * 1024,
            |_| {},
        )
        .expect("parallel download should succeed");

        assert_eq!(
            fs::read(&partial).expect("assembled download should be readable"),
            contents
        );
        remove_download_chunks(&partial);
        let _ = fs::remove_file(partial);
        server.join().expect("test server should exit");
    }

    #[test]
    fn removes_completed_download_fragments() {
        let runtime = temporary_directory("download-cleanup");
        let keep = runtime.join("runtime.ready");
        fs::write(&keep, RUNTIME_VERSION).expect("ready marker should be writable");
        for filename in [RUNTIME_ARCHIVE, SIDECAR_EXE] {
            let partial = runtime.join(filename).with_extension("part");
            fs::write(&partial, b"partial").expect("partial file should be writable");
            fs::write(download_chunk_path(&partial, 6), b"chunk")
                .expect("chunk file should be writable");
            fs::write(path_with_suffix(&partial, ".assembling"), b"assembly")
                .expect("assembling file should be writable");
        }

        cleanup_download_artifacts(&runtime);

        assert!(keep.is_file());
        assert_eq!(
            fs::read_dir(&runtime)
                .expect("runtime directory should remain readable")
                .count(),
            1
        );
        let _ = fs::remove_dir_all(runtime);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .manage(Sidecars::default())
        .manage(ServiceRuntime::new())
        .manage(RuntimeBootState::default())
        .manage(RuntimeSetupGate::default())
        .invoke_handler(tauri::generate_handler![
            get_runtime_boot_status,
            begin_runtime_setup,
            sidecar_status,
            get_service_endpoints,
            read_installer_audio_profile,
            clear_installer_audio_profile,
            stop_tts,
            restart_tts,
            shutdown_app,
            restart_app,
            get_app_info,
            open_runtime_folder,
            read_log_tail
        ])
        .setup(|app| {
            app.handle().plugin(
                tauri_plugin_log::Builder::default()
                    .level(log::LevelFilter::Info)
                    .build(),
            )?;
            let tts_port = match app
                .state::<ServiceRuntime>()
                .reserve_initial_ports()
            {
                Ok(port) => port,
                Err(error) => {
                    emit(app.handle(), "error", &error, 0);
                    log::error!("local service port reservation failed: {error}");
                    return Ok(());
                }
            };
            log::info!("Reserved main engine endpoint 127.0.0.1:{DEFAULT_BACKEND_PORT}");
            if tts_port == DEFAULT_TTS_PORT {
                log::info!("Reserved voice endpoint 127.0.0.1:{tts_port}");
            } else {
                log::warn!(
                    "Voice port {DEFAULT_TTS_PORT} is occupied; Mirid automatically selected 127.0.0.1:{}",
                    tts_port
                );
            }
            if let Err(error) = publish_service_endpoints(app.handle()) {
                emit(app.handle(), "error", &error, 0);
                log::error!("could not publish local service endpoints: {error}");
                return Ok(());
            }

            let handle = app.handle().clone();
            // Run runtime provisioning + sidecar startup off the main thread so the
            // window paints and can show download progress.
            std::thread::spawn(move || {
                if development_venv_enabled() {
                    emit(
                        &handle,
                        "starting",
                        "Starting Mirid from the local development environment.",
                        5,
                    );
                } else {
                    if let Err(err) = wait_for_runtime_setup(&handle) {
                        emit(
                            &handle,
                            "error",
                            &format!("Setup could not begin: {err}"),
                            0,
                        );
                        log::error!("runtime setup gate failed: {err}");
                        return;
                    }
                    if let Err(err) = ensure_runtime(&handle) {
                        emit(&handle, "error", &format!("Runtime setup failed: {err}"), 0);
                        log::error!("runtime setup failed: {err}");
                        return;
                    }
                }
                if let Err(err) = start_sidecars(&handle) {
                    emit(
                        &handle,
                        "error",
                        &format!("Failed to start services: {err}"),
                        0,
                    );
                    log::error!("failed to start sidecars: {err}");
                }
            });
            Ok(())
        })
        .on_window_event(|window, event| {
            if matches!(event, tauri::WindowEvent::Destroyed)
                && destroyed_window_owns_sidecars(window.label())
            {
                let app = window.app_handle();
                stop_all_sidecars(app);
                // The main window owns the desktop session. End the event loop
                // even if a standalone Settings window is still open.
                app.exit(0);
            }
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
