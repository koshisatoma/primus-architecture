use std::env;
use std::fs;
use std::net::UdpSocket;
use std::process::Command;

fn is_python_app(pid: u32) -> Result<bool, Box<dyn std::error::Error>> {
    let process_dir = format!("/proc/{pid}");
    let executable = fs::read_link(format!("{process_dir}/exe"))?;
    let is_python = executable
        .file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.starts_with("python"));
    let command_line = fs::read(format!("{process_dir}/cmdline"))?;
    let is_app = command_line
        .split(|byte| *byte == 0)
        .any(|arg| arg == b"app.py" || arg.ends_with(b"/app.py"));

    Ok(is_python && is_app)
}

fn main() {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "safety_rs".to_owned());
    let pid = args
        .next()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|pid| *pid > 1);
    let Some(pid) = pid else {
        eprintln!("Usage: {program} <app.py-pid>");
        std::process::exit(2);
    };

    match is_python_app(pid) {
        Ok(true) => {}
        Ok(false) => {
            eprintln!("[Rust Watchdog] Refusing PID {pid}: not a Python app.py process.");
            std::process::exit(2);
        }
        Err(error) => {
            eprintln!("[Rust Watchdog] Could not validate PID {pid}: {error}");
            std::process::exit(2);
        }
    }

    let socket = UdpSocket::bind("127.0.0.1:9119").expect("Failed to bind UDP 9119");
    let mut buf = [0; 32];
    println!("[Rust Watchdog] Watching app.py PID {pid} on 127.0.0.1:9119...");

    loop {
        let (amt, _src) = socket.recv_from(&mut buf).expect("Failed to receive");
        let msg = std::str::from_utf8(&buf[..amt]).unwrap_or("");

        if msg.trim() == "ESTOP" {
            match is_python_app(pid) {
                Ok(true) => {}
                Ok(false) => {
                    eprintln!(
                        "[Rust Watchdog] Refusing to signal PID {pid}: it is no longer app.py."
                    );
                    std::process::exit(1);
                }
                Err(error) => {
                    eprintln!("[Rust Watchdog] Could not revalidate PID {pid}: {error}");
                    std::process::exit(1);
                }
            }

            println!("\n[Rust Watchdog] 🚨 EMERGENCY STOP TRIGGERED! Stopping app.py PID {pid}...");
            let status = Command::new("kill")
                .arg("-TERM")
                .arg(pid.to_string())
                .status()
                .expect("Failed to invoke kill");
            if !status.success() {
                eprintln!("[Rust Watchdog] Failed to stop app.py PID {pid}.");
                std::process::exit(1);
            }
            println!("[Rust Watchdog] System halted.");
            break;
        } else {
            println!("[Rust Watchdog] Heartbeat received: {}", msg.trim());
        }
    }
}
