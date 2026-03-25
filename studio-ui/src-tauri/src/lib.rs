use std::{
  env,
  io::{Read, Write},
  net::{TcpListener, TcpStream},
  path::PathBuf,
  process::{Child, Command, Stdio},
  sync::Mutex,
  thread,
  time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use serde::Serialize;
use tauri::{AppHandle, Manager, RunEvent, State};

const STUDIO_HOST: &str = "127.0.0.1";
const SIDECAR_BASENAME: &str = "contextclaw-studio-daemon";
const STUDIO_PORT_ENV: &str = "CONTEXTCLAW_DESKTOP_PORT";
const STUDIO_TOKEN_HEADER: &str = "X-ContextClaw-Token";

struct DaemonRuntime {
  child: Child,
  port: u16,
  token: String,
}

#[derive(Default)]
struct StudioDaemonState {
  runtime: Mutex<Option<DaemonRuntime>>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct StudioInfo {
  base_url: String,
  port: u16,
}

fn build_token() -> String {
  let nanos = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_nanos();
  format!("studio-{}-{nanos}", std::process::id())
}

fn daemon_base_url(port: u16) -> String {
  format!("http://{STUDIO_HOST}:{port}")
}

fn make_http_request(port: u16, token: &str, method: &str, path: &str) -> Result<String, String> {
  let address = format!("{STUDIO_HOST}:{port}");
  let mut stream = TcpStream::connect(address).map_err(|error| error.to_string())?;
  let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
  let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
  let request = format!(
    "{method} {path} HTTP/1.1\r\nHost: {STUDIO_HOST}:{port}\r\n{STUDIO_TOKEN_HEADER}: {token}\r\nConnection: close\r\n\r\n"
  );
  stream
    .write_all(request.as_bytes())
    .map_err(|error| error.to_string())?;
  let mut response = String::new();
  stream
    .read_to_string(&mut response)
    .map_err(|error| error.to_string())?;
  Ok(response)
}

fn studio_status_healthy(port: u16, token: &str) -> bool {
  make_http_request(port, token, "GET", "/status")
    .map(|response| response.contains("200 OK") && response.contains("\"status\":\"ok\""))
    .unwrap_or(false)
}

fn wait_for_studio_status(port: u16, token: &str, timeout: Duration) -> Result<(), String> {
  let deadline = Instant::now() + timeout;
  while Instant::now() < deadline {
    if studio_status_healthy(port, token) {
      return Ok(());
    }
    thread::sleep(Duration::from_millis(150));
  }
  Err(format!(
    "Timed out waiting for Studio daemon at {}",
    daemon_base_url(port)
  ))
}

fn request_daemon_shutdown(port: u16, token: &str) -> Result<(), String> {
  let response = make_http_request(port, token, "POST", "/shutdown")?;
  if response.contains("200 OK") {
    return Ok(());
  }
  Err(format!("Studio daemon shutdown request failed for {}", daemon_base_url(port)))
}

fn reserve_port() -> Result<u16, String> {
  let listener = TcpListener::bind((STUDIO_HOST, 0)).map_err(|error| error.to_string())?;
  listener
    .local_addr()
    .map(|address| address.port())
    .map_err(|error| error.to_string())
}

fn sidecar_file_name() -> String {
  format!("{SIDECAR_BASENAME}{}", env::consts::EXE_SUFFIX)
}

fn sidecar_candidates(app: &AppHandle) -> Vec<PathBuf> {
  let file_name = sidecar_file_name();
  let mut candidates = Vec::new();

  if let Ok(current_exe) = env::current_exe() {
    if let Some(exe_dir) = current_exe.parent() {
      candidates.push(exe_dir.join(&file_name));
    }
  }

  if let Ok(resource_dir) = app.path().resource_dir() {
    candidates.push(resource_dir.join(&file_name));
    #[cfg(target_os = "macos")]
    if let Some(contents_dir) = resource_dir.parent() {
      candidates.push(contents_dir.join("MacOS").join(&file_name));
    }
  }

  candidates
}

fn resolve_sidecar_path(app: &AppHandle) -> Result<PathBuf, String> {
  sidecar_candidates(app)
    .into_iter()
    .find(|path| path.is_file())
    .ok_or_else(|| "Unable to locate the bundled Studio daemon sidecar".to_string())
}

fn stop_studio_daemon(app: &AppHandle) {
  let state = app.state::<StudioDaemonState>();
  let runtime = {
    let mut guard = state.runtime.lock().expect("studio daemon mutex poisoned");
    guard.take()
  };

  if let Some(mut runtime) = runtime {
    if let Err(error) = request_daemon_shutdown(runtime.port, &runtime.token) {
      log::warn!("failed to request Studio daemon shutdown: {error}");
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
      match runtime.child.try_wait() {
        Ok(Some(_)) => return,
        Ok(None) => thread::sleep(Duration::from_millis(150)),
        Err(error) => {
          log::warn!("failed to inspect Studio daemon shutdown state: {error}");
          break;
        }
      }
    }
    if let Err(error) = runtime.child.kill() {
      log::warn!("failed to stop Studio daemon forcefully: {error}");
    }
    if let Err(error) = runtime.child.wait() {
      log::warn!("failed waiting for Studio daemon shutdown: {error}");
    }
  }
}

fn configured_port() -> Result<Option<u16>, String> {
  match env::var(STUDIO_PORT_ENV) {
    Ok(value) => {
      let port = value
        .parse::<u16>()
        .map_err(|error| format!("Invalid {STUDIO_PORT_ENV} value `{value}`: {error}"))?;
      Ok(Some(port))
    }
    Err(_) => Ok(None),
  }
}

fn ensure_studio_daemon(app: &AppHandle) -> Result<(), String> {
  let state = app.state::<StudioDaemonState>();
  {
    let mut guard = state.runtime.lock().expect("studio daemon mutex poisoned");
    if let Some(runtime) = guard.as_mut() {
      match runtime.child.try_wait() {
        Ok(None) => {
          let port = runtime.port;
          let token = runtime.token.clone();
          drop(guard);
          return wait_for_studio_status(port, &token, Duration::from_secs(10));
        }
        Ok(Some(_)) => {
          *guard = None;
        }
        Err(error) => {
          *guard = None;
          log::warn!("failed to inspect Studio daemon state: {error}");
        }
      }
    }
  }

  let sidecar_path = resolve_sidecar_path(app)?;
  let requested_port = configured_port()?;
  let attempts = if requested_port.is_some() { 1 } else { 5 };

  for _ in 0..attempts {
    let port = match requested_port {
      Some(port) => port,
      None => reserve_port()?,
    };
    let token = build_token();
    let mut child = Command::new(&sidecar_path)
      .env("CONTEXTCLAW_STUDIO_HOST", STUDIO_HOST)
      .env("CONTEXTCLAW_STUDIO_PORT", port.to_string())
      .env("CONTEXTCLAW_STUDIO_TOKEN", &token)
      .stdin(Stdio::null())
      .stdout(Stdio::null())
      .stderr(Stdio::null())
      .spawn()
      .map_err(|error| format!("Failed to start Studio daemon: {error}"))?;

    match wait_for_studio_status(port, &token, Duration::from_secs(10)) {
      Ok(()) => {
        let mut guard = state.runtime.lock().expect("studio daemon mutex poisoned");
        *guard = Some(DaemonRuntime { child, port, token });
        return Ok(());
      }
      Err(error) => {
        let _ = child.kill();
        let _ = child.wait();
        if requested_port.is_some() {
          return Err(error);
        }
      }
    }
  }

  Err("Failed to start Studio daemon after multiple attempts".to_string())
}

#[tauri::command]
fn studio_info(state: State<'_, StudioDaemonState>) -> Result<StudioInfo, String> {
  let guard = state.runtime.lock().expect("studio daemon mutex poisoned");
  let runtime = guard
    .as_ref()
    .ok_or_else(|| "Studio daemon is not running".to_string())?;
  Ok(StudioInfo {
    base_url: daemon_base_url(runtime.port),
    port: runtime.port,
  })
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  let app = tauri::Builder::default()
    .manage(StudioDaemonState::default())
    .invoke_handler(tauri::generate_handler![studio_info])
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      ensure_studio_daemon(&app.handle()).map_err(std::io::Error::other)?;
      Ok(())
    })
    .build(tauri::generate_context!())
    .expect("error while building tauri application");

  app.run(|app_handle, event| {
    if matches!(event, RunEvent::Exit | RunEvent::ExitRequested { .. }) {
      stop_studio_daemon(app_handle);
    }
  });
}
