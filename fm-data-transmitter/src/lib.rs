use data2sound::encode;
use std::env;
use std::fs;
use std::process::Command;

fn main() {
    let args: Vec<String> = env::args().collect();
    let filename: &str = args[1].as_str();
    send_file(filename);
}

pub fn send_file(filename: &str) {
    let sound_file: &str = &format!("{filename}.wav");

    convert_file_to_sound(filename, sound_file);

    send_sound_file(sound_file);

    fs::remove_file(sound_file);
    fs::remove_file(filename);
}

pub fn convert_file_to_sound(filename: &str, sound_file: &str) {
    encode(filename, sound_file).unwrap()
}

pub fn send_sound_file(sound_file: &str) {
    let command = format!(
        "{}/fm-data-transmitter/fm_transmitter",
        env::current_dir()
            .expect("Couldn't obtain current directory path")
            .display()
    );

    let output = Command::new(command)
        .arg(sound_file)
        .output()
        .expect("failed to execute process");

    println!("status: {}", output.status);
    println!("stdout: {}", String::from_utf8_lossy(&output.stdout));
    println!("stderr: {}", String::from_utf8_lossy(&output.stderr));
}
