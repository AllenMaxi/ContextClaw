use std::{
  env,
  io::{Read, Write},
  net::TcpStream,
  path::PathBuf,
  process::{Child, Command, Stdio},
  sync::Mutex,
  thread,
  time::{Duration, Instant},
};

use tauri::{AppHandle, Manager, RunEvent};

const STUDIO_HOST: &str = "127.0.0.1";
const STUDIO_PORT: u16 = 8765;
const SIDECAR_BASENAME: &str = "contextclaw-studio-daemon";

#[derive(Default)]
struct StudioDaemonState {
  child: Mutex<Option<Child>>,
}

fn daemon_url() -> String {
  format!("http://{STUDIO_HOST}:{STUDIO_PORT}/status")
}

fn studio_status_healthy() -> bool {
  let address = format!("{STUDIO_HOST}:{STUDIO_PORT}");
  let mut stream = match TcpStream::connect(address) {
    Ok(stream) => stream,
    Err(_) => return false,
  };
  let _ = stream.set_read_timeout(Some(Duration::from_millis(500)));
  let _ = stream.set_write_timeout(Some(Duration::from_millis(500)));
  let request = format!(
    "GET /status HTTP/1.1\r\nHost: {STUDIO_HOST}:{STUDIO_PORT}\r\nConnection: close\r\n\r\n"
  );
  if stream.write_all(request.as_bytes()).is_err() {
    return false;
  }
  let mut response = String::new();
  if stream.read_to_string(&mut response).is_err() {
    return false;
  }
  response.contains("200 OK") && response.contains("\"status\":\"ok\"")
}

fn wait_for_studio_status(timeout: Duration) -> Result<(), String> {
  let deadline = Instant::now() + timeout;
  while Instant::now() < deadline {
    if studio_status_healthy() {
      return Ok(());
    }
    thread::sleep(Duration::from_millis(150));
  }
  Err(format!("Timed out waiting for Studio daemon at {}", daemon_url()))
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
  let mut guard = state.child.lock().expect("studio daemon mutex poisoned");
  if let Some(mut child) = guard.take() {
    if let Err(error) = child.kill() {
      log::warn!("failed to stop Studio daemon: {error}");
    }
    if let Err(error) = child.wait() {
      log::warn!("failed waiting for Studio daemon shutdown: {error}");
    }
  }
}

fn ensure_studio_daemon(app: &AppHandle) -> Result<(), String> {
  if studio_status_healthy() {
    log::info!("Using existing Studio daemon at {}", daemon_url());
    return Ok(());
  }

  let state = app.state::<StudioDaemonState>();
  {
    let mut guard = state.child.lock().expect("studio daemon mutex poisoned");
    if let Some(child) = guard.as_mut() {
      match child.try_wait() {
        Ok(None) => {
          drop(guard);
          return wait_for_studio_status(Duration::from_secs(10));
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
  log::info!("Starting Studio daemon sidecar from {}", sidecar_path.display());
  let child = Command::new(&sidecar_path)
    .env("CONTEXTCLAW_STUDIO_HOST", STUDIO_HOST)
    .env("CONTEXTCLAW_STUDIO_PORT", STUDIO_PORT.to_string())
    .stdin(Stdio::null())
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .spawn()
    .map_err(|error| format!("Failed to start Studio daemon: {error}"))?;

  {
    let mut guard = state.child.lock().expect("studio daemon mutex poisoned");
    *guard = Some(child);
  }

  if let Err(error) = wait_for_studio_status(Duration::from_secs(10)) {
    stop_studio_daemon(app);
    return Err(error);
  }
  Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  let app = tauri::Builder::default()
    .manage(StudioDaemonState::default())
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
