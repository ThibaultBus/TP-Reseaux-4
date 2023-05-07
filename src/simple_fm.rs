//! FM radio example with hard-coded params that outputs raw audio
//! data to stdout just like the original rtl_fm.
//!
//! Can also read raw data from a file instead of a real rtl-sdr device by
//! setting READ_FROM_FILE to true, which can be a good way to verify that
//! audio output is working.
//!
//! Example command to run the program and output audio with `play` (must be installed):
//! cargo run --example simple_fm | play -r 32k -t raw -e s -b 16 -c 1 -V1 -

use core::alloc::Layout;
use ctrlc;
use log::info;
use num_complex::Complex;
use rtlsdr_rs::{error::Result, RtlSdr, DEFAULT_BUF_LENGTH};
use std::alloc::alloc_zeroed;
use std::f64::consts::PI;
use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;

// Radio and demodulation config
const FREQUENCY: u32 = 100_000_000; // Frequency in Hz
const SAMPLE_RATE: u32 = 170_000; // Demodulation sample rate, 170kHz
const RATE_RESAMPLE: u32 = 32_000; // Output sample rate, 32kHz

// Switch to read raw data from file instead of real device, and what file to read from.
// Setting this to true can be a quick way to verify that the program and audio output is working.
const READ_FROM_FILE: bool = false;
const INPUT_FILE_PATH: &str = "capture.bin";

pub fn main() {
    // Printing to stdout will break audio output, so use this to log to stderr instead
    stderrlog::new().verbosity(log::Level::Info).init().unwrap();

    // Shutdown flag that is set true when ctrl-c signal caught
    static SHUTDOWN: AtomicBool = AtomicBool::new(false);
    ctrlc::set_handler(|| {
        SHUTDOWN.swap(true, Ordering::Relaxed);
    })
    .unwrap();

    // Get radio and demodulation settings for given frequency and sample rate
    let (radio_config, demod_config) = optimal_settings(FREQUENCY, SAMPLE_RATE);

    // Check if configured to use real device or read from file
    if !READ_FROM_FILE {
        // Real device! Will use two threads, one to handle the SDR and one for demodulation and output

        // Channel to pass receive data from receiver thread to processor thread
        let (tx, rx) = mpsc::channel();

        // Spawn thread to receive data from Radio
        let receive_thread = thread::spawn(|| receive(&SHUTDOWN, radio_config, tx));
        // Spawn thread to process data and output to stdout
        let process_thread = thread::spawn(|| process(&SHUTDOWN, demod_config, rx));

        // Wait for threads to finish
        process_thread.join().unwrap();
        receive_thread.join().unwrap();
    } else {
        // Read raw data from file instead of real device
        use std::fs::File;
        use std::io::prelude::*;
        let mut f = File::open(INPUT_FILE_PATH).expect("failed to open file");
        let mut buf = [0_u8; DEFAULT_BUF_LENGTH];
        let mut demod = Demod::new(demod_config);
        loop {
            // Check if shutdown signal received
            if SHUTDOWN.load(Ordering::Relaxed) {
                break;
            }
            // Read chunk of file  data into buf
            let n = f.read(&mut buf[..]).expect("failed to read");
            // Demodulate data from file
            let result = demod.demodulate(buf.to_vec());
            // Output resulting audio data to stdout

            // output(result);
        }
    }
}

/// Thread to open SDR device and send received data to the demod thread until
/// SHUTDOWN flag is set to true.
fn receive(shutdown: &AtomicBool, radio_config: RadioConfig, tx: Sender<Vec<u8>>) {
    // Open device
    let mut sdr = RtlSdr::open(0).expect("Failed to open device");
    // Config receiver
    config_sdr(
        &mut sdr,
        radio_config.capture_freq,
        radio_config.capture_rate,
    )
    .unwrap();

    info!("Tuned to {} Hz.\n", sdr.get_center_freq());
    info!(
        "Buffer size: {}ms",
        1000.0 * 0.5 * DEFAULT_BUF_LENGTH as f32 / radio_config.capture_rate as f32
    );
    info!("Sampling at {} S/s", sdr.get_sample_rate());

    info!("Reading samples in sync mode...");
    loop {
        // Check if SHUTDOWN flag is true and break out of the loop if so
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        // Allocate a buffer to store received data
        let mut buf: Box<[u8; DEFAULT_BUF_LENGTH]> = alloc_buf();
        // Receive data from SDR device
        let n = sdr.read_sync(&mut *buf);
        if n.is_err() {
            info!("Read error: {:#?}", n);
            break;
        }
        let len = n.unwrap();
        if len < DEFAULT_BUF_LENGTH {
            info!("Short read ({:#?}), samples lost, exiting!", len);
            break;
        }
        // Send received data through the channel to the processor thread
        tx.send(buf.to_vec());
    }
    // Shut down the device and exit
    info!("Close");
    sdr.close().unwrap();
}

/// Thread to process received data and output it to stdout
fn process(shutdown: &AtomicBool, demod_config: DemodConfig, rx: Receiver<Vec<u8>>) {
    // Create and configure demodulation struct
    let mut demod = Demod::new(demod_config);
    info!("Oversampling input by: {}x", demod.config.downsample);
    info!("Output at {} Hz", demod.config.rate_in);
    info!("Output scale: {}", demod.config.output_scale);
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        // Wait for data from the channel
        let buf = rx.recv().unwrap();
        // Demodulate data
        let result = demod.demodulate(buf);
        // Output audio data to stdout
        output(result);
    }
}

/// Radio configuration produced by `optimal_settings`
struct RadioConfig {
    capture_freq: u32,
    capture_rate: u32,
}

/// Demodulation configuration produced by `optimal_settings`
struct DemodConfig {
    rate_in: u32,       // Rate in Hz
    rate_out: u32,      // Rate in Hz
    rate_resample: u32, // Rate in Hz
    downsample: u32,
    output_scale: u32,
}

/// Determine the optimal radio and demodulation configurations for given
/// frequency and sample rate.
fn optimal_settings(freq: u32, rate: u32) -> (RadioConfig, DemodConfig) {
    let downsample = (1_000_000 / rate) + 1;
    // let downsample = 20;
    info!("downsample: {}", downsample);
    let capture_rate = downsample * rate;
    info!("rate_in: {} capture_rate: {}", rate, capture_rate);
    // Use offset-tuning
    let capture_freq = 100_000_000; //(freq + capture_rate) / 4;
    info!("capture_freq: {}", capture_freq);
    let mut output_scale = (1 << 15) / (128 * downsample);
    if output_scale < 1 {
        output_scale = 1;
    }
    (
        RadioConfig {
            capture_freq: capture_freq,
            capture_rate: capture_rate,
        },
        DemodConfig {
            rate_in: SAMPLE_RATE,
            rate_out: SAMPLE_RATE,
            rate_resample: RATE_RESAMPLE,
            downsample: downsample,
            output_scale: output_scale,
        },
    )
}

/// Configure the SDR device for a given receive frequency and sample rate.
fn config_sdr(sdr: &mut RtlSdr, freq: u32, rate: u32) -> Result<()> {
    // Use auto-gain
    sdr.set_tuner_gain(rtlsdr_rs::TunerGain::Auto)?;
    // Disable bias-tee
    sdr.set_bias_tee(false)?;
    // Reset the endpoint before we try to read from it (mandatory)
    sdr.reset_buffer()?;
    // Set the frequency
    sdr.set_center_freq(freq)?;
    // Set sample rate
    sdr.set_sample_rate(rate)?;
    Ok(())
}

/// State data for demodulation
struct Demod {
    config: DemodConfig,
    prev_index: usize,
    now_lpr: i32,
    prev_lpr_index: i32,
    lp_now: Complex<i32>,
    demod_pre: Complex<i32>,
}

/// Demodulation functions
impl Demod {
    fn new(config: DemodConfig) -> Self {
        Demod {
            config: config,
            prev_index: 0,
            now_lpr: 0,
            prev_lpr_index: 0,
            lp_now: Complex::new(0, 0),
            demod_pre: Complex::new(0, 0),
        }
    }

    /// Performs the entire demodulation process, given a vector of raw received bytes
    /// returns a vector of signed 16-bit audio data.
    fn demodulate(&mut self, mut buf: Vec<u8>) -> Vec<i16> {
        buf = Demod::rotate_90(buf);
        let buf_signed: Vec<i16> = buf.iter().map(|val| *val as i16 - 127).collect();
        let complex = buf_to_complex(buf_signed);
        // low-pass filter to downsample to our desired sample rate
        // let lowpassed = self.high_pass_complex(complex);
        // dbg!(&lowpassed);

        // Demodulate FM signal
        // let demodulated = self.fm_demod(lowpassed);
        let demodulated = self.fm_demod(complex);
        // dbg!(&demodulated);

        // Resample and return result
        let output = self.low_pass_real(demodulated);
        // dbg!(&output);
        // panic!();
        output
    }

    /// Performs a 90-degree rotation in the complex plane on a vector of bytes
    /// and returns the resulting vector.
    /// Data is assumed to be pairs of real and imaginary components.
    fn rotate_90(mut buf: Vec<u8>) -> Vec<u8> {
        /* 90 rotation is 1+0j, 0+1j, -1+0j, 0-1j
        or [0, 1, -3, 2, -4, -5, 7, -6] */
        let mut tmp: u8;
        for i in (0..buf.len()).step_by(8) {
            /* uint8_t negation = 255 - x */
            tmp = 255 - buf[i + 3];
            buf[i + 3] = buf[i + 2];
            buf[i + 2] = tmp;

            buf[i + 4] = 255 - buf[i + 4];
            buf[i + 5] = 255 - buf[i + 5];

            tmp = 255 - buf[i + 6];
            buf[i + 6] = buf[i + 7];
            buf[i + 7] = tmp;
        }
        buf
    }

    /// Applies a low-pass filter on a vector of complex values
    fn low_pass_complex(&mut self, buf: Vec<Complex<i32>>) -> Vec<Complex<i32>> {
        let mut res = vec![];
        for orig in 0..buf.len() {
            self.lp_now += buf[orig];

            self.prev_index += 1;
            // if self.prev_index < self.config.downsample as usize {
            //     continue;
            // }

            res.push(self.lp_now);
            self.lp_now = Complex::new(0, 0);
            self.prev_index = 0;
        }
        res
    }

    /// Applies a high-pass filter on a vector of complex values
    fn high_pass_complex(&mut self, buf: Vec<Complex<i32>>) -> Vec<Complex<i32>> {
        let mut res = vec![];
        dbg!(&buf.len());
        for orig in 0..buf.len() {
            self.lp_now += buf[orig];

            self.prev_index += 1;
            // if self.prev_index > self.config.downsample as usize {
            //     continue;
            // }

            res.push(self.lp_now);
            self.lp_now = Complex::new(0, 0);
            self.prev_index = 0;
        }
        res
    }

    /// Performs FM demodulation on a vector of complex input data
    fn fm_demod(&mut self, buf: Vec<Complex<i32>>) -> Vec<i16> {
        assert!(buf.len() > 1);
        let mut result = vec![];

        let mut pcm = Demod::polar_discriminant(buf[0], self.demod_pre);
        result.push(pcm as i16);
        for i in 1..buf.len() {
            pcm = Demod::polar_discriminant_fast(buf[i], buf[i - 1]);
            result.push(pcm as i16);
        }
        self.demod_pre = buf.last().copied().unwrap();
        result
    }

    /// Find the polar discriminant for a pair of complex values using real atan2 function
    fn polar_discriminant(a: Complex<i32>, b: Complex<i32>) -> i32 {
        let c = a * b.conj();
        let angle = f64::atan2(c.im as f64, c.re as f64);
        (angle / PI * (1 << 14) as f64) as i32
    }

    /// Find the polar discriminant for a pair of complex values using a fast atan2 approximation
    fn polar_discriminant_fast(a: Complex<i32>, b: Complex<i32>) -> i32 {
        let c = a * b.conj();
        Demod::fast_atan2(c.im, c.re)
    }

    /// Fast atan2 approximation
    fn fast_atan2(y: i32, x: i32) -> i32 {
        // Pre-scaled for i16
        // pi = 1 << 14
        let pi4 = 1 << 12;
        let pi34 = 3 * (1 << 12);
        if x == 0 && y == 0 {
            return 0;
        }
        let mut yabs = y;
        if yabs < 0 {
            yabs = -yabs;
        }
        let angle;
        if x >= 0 {
            angle = pi4 - (pi4 as i64 * (x - yabs) as i64) as i32 / (x + yabs);
        } else {
            angle = pi34 - (pi4 as i64 * (x + yabs) as i64) as i32 / (yabs - x);
        }
        if y < 0 {
            return -angle;
        }
        return angle;
    }

    /// Applies a low-pass filter to a vector of real-valued data
    fn low_pass_real(&mut self, buf: Vec<i16>) -> Vec<i16> {
        let mut result = vec![];
        // Simple square-window FIR
        let slow = self.config.rate_resample;
        let fast = self.config.rate_out;
        let mut i = 0;
        while i < buf.len() {
            self.now_lpr += buf[i] as i32;
            i += 1;
            self.prev_lpr_index += slow as i32;
            if self.prev_lpr_index < fast as i32 {
                continue;
            }
            result.push((self.now_lpr / ((fast / slow) as i32)) as i16);
            self.prev_lpr_index -= fast as i32;
            self.now_lpr = 0;
        }
        result
    }
}

/// Write a vector of i16 values to stdout
fn output(buf: Vec<i16>) {
    use std::{mem, slice};
    // let mut out = std::io::stdout();
    let mut out = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open("tmp/packets/recv/input.wav")
        .unwrap();
    let slice_u8: &[u8] = unsafe {
        slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len() * mem::size_of::<i16>())
    };
    out.write_all(slice_u8);
    out.flush();
}

/// Convert a vector of i16 complex components (real and imaginary) to a vector of i32 Complex values
fn buf_to_complex(buf: Vec<i16>) -> Vec<Complex<i32>> {
    buf
        // get overlapping windows of size 2
        .windows(2)
        // Step by 2 since we don't actually want overlapping windows
        .step_by(2)
        // Convert consecutive values to a single complex
        .map(|w| Complex::new(w[0] as i32, w[1] as i32))
        .collect()
}
/// Allocate a buffer on the heap
fn alloc_buf<T>() -> Box<T> {
    let layout: Layout = Layout::new::<T>();
    // TODO move to using safe code once we can allocate an array directly on the heap.
    unsafe {
        let ptr = alloc_zeroed(layout) as *mut T;
        Box::from_raw(ptr)
    }
}
