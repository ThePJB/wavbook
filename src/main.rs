extern crate riff_wave;

mod random;
mod vector;

use random::*;

use std::f32::consts::PI;
use std::fs::File;
use std::io::BufWriter;

use riff_wave::{WaveWriter, WriteResult, WaveReader};

use std::io::Read;

pub fn khash(mut state: u32) -> u32 {
    state = (state ^ 2747636419).wrapping_mul(2654435769);
    state = (state ^ (state >> 16)).wrapping_mul(2654435769);
    state = (state ^ (state >> 16)).wrapping_mul(2654435769);
    state
}
pub fn krand(seed: u32) -> f32 {
    (khash(seed)&0x00000000FFFFFFFF) as f32 / 4294967295.0
}

const SAMPLE_RATE: u32 = 44100;


fn main() {

}

#[test]
fn bassslines() -> WriteResult<()> {		
	const FREQUENCY: f32 = 2.0 * PI * 50.0; // radian per second

	let file = File::create("randombassline.wav")?;
	let writer = BufWriter::new(file);

	let mut wave_writer = WaveWriter::new(1, SAMPLE_RATE, 16, writer)?;

    let beat_length = 0.5;
    let beats_per_bar = 4;
    let repeats = 4;
    let bars_per_repeat = 1;
    let total_bars = repeats * bars_per_repeat;
    let length = beats_per_bar as f64 * beat_length as f64 * total_bars as f64 * repeats as f64;
    dbg!(length);
    let N = (length * SAMPLE_RATE as f64).round() as u64;
    let samples_per_beat = (beat_length * SAMPLE_RATE as f64).round() as u64;
    let samples_per_repeat = N / repeats as u64;
    let samples_per_bar = samples_per_repeat / bars_per_repeat as u64;
    let total_beats = total_bars * beats_per_bar;



    let seed: u32 = 69;


    let mut magnitude = 0.1;
    let mut phase = 0.0;

    
	for n in 0..N {
        let n_repeat = n / samples_per_repeat;
        let n_bar = n_repeat % samples_per_bar;
        let n_beat = n_bar % samples_per_beat;

        let beat_num = N / samples_per_beat;
        let bar_num = N / samples_per_bar;

        // still have to define the specific pattern
        // could do it with a random seed

        let subdivisions_per_beat = 4;
        let total_subdivisions = total_beats * subdivisions_per_beat;
        let subs_per_repeat = total_subdivisions / repeats;
        let samples_per_sub = samples_per_beat / subdivisions_per_beat as u64;
        let n_sub = n_beat % (samples_per_sub);
        let sub_num = n / samples_per_sub;
        let sub_num_of_repeat = sub_num % subs_per_repeat;

        // n sub would be synth input
        // println!("n: {}, sub num: {}", n, sub_num);
        let sub_seed = seed.wrapping_add((sub_num_of_repeat as u32).wrapping_mul(123712377));
        let roll = krand(sub_seed);
        // println!("roll: {}", roll);
        let amp = if roll > 0.5 {
            // println!("yes");
            1.0
        } else {
            // println!("no");
            0.0
        };


        let f = 100.0;
        phase += f * 2.0 * PI / SAMPLE_RATE as f32;

        let x = amp * magnitude * phase.sin();

        let sample = (x * i16::MAX as f32) as i16;

        wave_writer.write_sample_i16(sample)?;
	}

	Ok(())
}







#[test]
fn write_wave() -> WriteResult<()> {		
	const FREQUENCY: f32 = 2.0 * PI * 50.0; // radian per second

	let file = File::create("hello.wav")?;
	let writer = BufWriter::new(file);

	let mut wave_writer = WaveWriter::new(1, SAMPLE_RATE, 16, writer)?;


    let mut magnitude = 0.1;
    let mut phase = 0.0;


	for n in 0..SAMPLE_RATE*4 {

        let f = 50.0;
        phase += f * 2.0 * PI / SAMPLE_RATE as f32;
        if krand(n) < 0.1 {
            phase += f * 2.0 * PI / SAMPLE_RATE as f32;
        }
        // magnitude *= 0.9999;

        let x = magnitude * phase.sin();

        // let x = if x > 0.0 {
        //     1.0 - (1.0 - x)*(1.0-x)
        // } else {
        //     -(1.0 - (1.0 + x)*(1.0 + x))
        // };

		let sample = (x * i16::MAX as f32) as i16;

		wave_writer.write_sample_i16(sample)?;
	}

	Ok(())
}

#[test]
fn decimate_white() -> WriteResult<()> {		
	let file = File::create("decimatewhite.wav")?;
	let writer = BufWriter::new(file);

	let mut wave_writer = WaveWriter::new(1, SAMPLE_RATE, 16, writer)?;

    let d = 32;
    let nd = 64;


	for n in 0..SAMPLE_RATE*4 {
        let mut acc = 0.0;
        for di in 0..nd {
            acc += krand(khash(di).wrapping_add(khash(n/(1 + d*di))));  // dont divide by zero here
        }
        acc /= nd as f32;

        wave_writer.write_sample_i16((acc * i16::MAX as f32) as i16)?;
	}

	Ok(())
}

#[test]
fn write_wave_biphase() -> WriteResult<()> {		
	const FREQUENCY: f32 = 2.0 * PI * 50.0; // radian per second

	let file = File::create("biphase.wav")?;
	let writer = BufWriter::new(file);

	let mut wave_writer = WaveWriter::new(1, SAMPLE_RATE, 16, writer)?;


    let mut magnitude = 0.1;
    let mut phase1 = 0.0;
    let mut phase2 = 0.0;


	for n in 0..SAMPLE_RATE*4 {

        let f = 110.0;
        if krand(n) < 0.9999 {
            phase1 += f * 2.0 * PI / SAMPLE_RATE as f32;
        }
        if krand(khash(n)) < 0.9998 {
            phase2 += f * 2.0 * PI / SAMPLE_RATE as f32;
        }

        let x = magnitude * (phase1.sin() + phase2.sin())/0.5;

		let sample = (x * i16::MAX as f32) as i16;

		wave_writer.write_sample_i16(sample)?;
	}

	Ok(())
}

#[test]
fn write_wave2() -> WriteResult<()> {		
	const FREQUENCY: f32 = 2.0 * PI * 50.0; // radian per second

	let file = File::create("hello2.wav")?;
	let writer = BufWriter::new(file);

	let mut wave_writer = WaveWriter::new(1, SAMPLE_RATE, 16, writer)?;


    let mut magnitude = 0.1;
    let mut phase: f32 = 0.0;


	for n in 0..SAMPLE_RATE*4 {

        let f = 440.0;
        let x = phase.sin();
        phase += if x.abs() > 0.7 {
            2.0 * f * 2.0 * PI / SAMPLE_RATE as f32
        } else {
            f * 2.0 * PI / SAMPLE_RATE as f32
        };

		let sample = (magnitude * x * i16::MAX as f32) as i16;

		wave_writer.write_sample_i16(sample)?;
	}

	Ok(())
}

// could have a 2d plot where you set the threshold and amount

// what other sound could be made

// ya t hat sounds pretty decent

// what about randomized detune where the detune is random walking or they both have a chance to drop 1 phase
fn repeating_impulse(n: i32, N: i32) -> f32 {
    if n % N == 0 {
        1.0
    } else {
        0.0
    }
}

fn impulse_decay_linear(n: i32, N: i32, d: f32) -> f32 {
    let rem = n % N;
    // 0.0: 1.0
    // else: decay by how many, eg to power of d
    (1.0 - d * rem as f32).max(0.0)

}

fn impulse_decay(n: i32, N: i32, d: f32) -> f32 {
    let rem = n % N;
    // 0.0: 1.0
    // else: decay by how many, eg to power of d
    1.0*(1.0 - d).powf(rem as f32)

}

fn sample_vec_i32(vec: &Vec<f32>, n: i32) -> f32 {
    if n < 0 || n >= vec.len() as i32 {
        return 0.0;
    }
    vec[n as usize]
}

fn sin_exciter(n: i32, omega: f32) -> f32 {
    let env = impulse_decay(n, 20000, 0.0001);
    let phase = omega * n as f32;
    env * phase.sin()
}

fn white_noise(n: i32) -> f32 {
    krand(n as u32)
}

fn part_cycle<F>(f: F, full: i32, part: i32, n: i32) -> f32
where
    F: Fn(i32) -> f32,
{
    if n % full < part {
        f(n)
    } else {
        0.0
    }
}

#[test]
fn test_impulse_decay() -> WriteResult<()> {		
    let mut samples = Vec::new();
	for n in 0..SAMPLE_RATE as i32 *4 {
        samples.push(0.1 * sin_exciter(n, 25.0));
	}
    
    let file = File::create("waveguide.wav")?;
	let writer = BufWriter::new(file);
	let mut wave_writer = WaveWriter::new(1, SAMPLE_RATE, 16, writer)?;
    for s in samples {
        wave_writer.write_sample_i16((s * i16::MAX as f32) as i16)?;
    }

	Ok(())
}

// he be usin dat pink noise exciter
// pitch inverse of delay length
// negative feedback: flute - cancelling of even harmonics

#[test]
fn make_white_noise() {
    let g = |n| 0.1*white_noise(n);
    write_wav("white.wav", make_samples(g, SAMPLE_RATE as i32 * 4));
}

#[test]
fn make_mult_noise() {
    let g = |n| 0.1*white_noise(n)*white_noise(n+1094781247)*white_noise(n+2131237)*white_noise(n+315123667);
    write_wav("mult.wav", make_samples(g, SAMPLE_RATE as i32 * 4));
}

#[test]
fn digital_waveguide_synthesis() {		

    // let g = |n| amp*sin_exciter(n, 25.0);
    // let g = |n| amp*impulse_decay(n, 20000, 0.001);
    let g = |n| part_cycle(white_noise, 21000, 6000, n);

    write_wav("dws_excite.wav", make_samples(g, SAMPLE_RATE as i32 * 4));
    write_wav("dws_fdbk1.wav", make_samples_feedback(g, SAMPLE_RATE as i32 * 4, -0.85,2669));
    write_wav("dws_fdbk2.wav", make_samples_feedback(g, SAMPLE_RATE as i32 * 4, -0.5, 420));
    write_wav("dws_fdbka3.wav", make_samples_feedback(g, SAMPLE_RATE as i32 * 4, 0.75,1000));

	// for n in 0..SAMPLE_RATE as i32 *4 {
    //     let x = sin_exciter(n-d, 25.0);
    //     let feedback = a * sample_vec_i32(&samples, n-d);
    //     samples.push(amp * (x + feedback));
	// }
    
    // write_wav("waveguide.wav", samples);
}

fn make_samples_feedback<F>(f: F, N: i32, a: f32, d: i32) -> Vec<f32>
where
    F: Fn(i32) -> f32,
{
    let mut samples = Vec::new();
    for n in 0..N {
        samples.push(f(n-d) + a*sample_vec_i32(&samples, n-d))
    }
    samples
}

fn make_samples<F>(f: F, N: i32) -> Vec<f32>
where
    F: Fn(i32) -> f32,
{
    (0..N).map(|n| f(n)).collect()
}

fn write_wav(outfile: &str, samples: Vec<f32>) {
    let file = File::create(outfile).unwrap();
	let writer = BufWriter::new(file);
	let mut wave_writer = WaveWriter::new(1, SAMPLE_RATE, 16, writer).unwrap();
    for s in samples {
        wave_writer.write_sample_i16((s * i16::MAX as f32) as i16).unwrap();
    }
}