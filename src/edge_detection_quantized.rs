use std::time::Instant;

use image::ImageBuffer;
use num_integer::Roots;
use rayon::prelude::*;
use ripple::common::ct_lut_eval_quantized_no_gen;
use tfhe::{
    integer::{
        gen_keys_radix, wopbs::*, IntegerCiphertext, IntegerRadixCiphertext, RadixCiphertext,
        ServerKey,
    },
    shortint::parameters::{
        parameters_wopbs_message_carry::WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS, Degree,
        PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    },
};

fn square_matmul(
    mat: &[RadixCiphertext],
    shifts: &[i32],
    nb_blocks: usize,
    server_key: &ServerKey,
) -> Vec<RadixCiphertext> {
    let size = mat.len().sqrt();
    let mut prod: Vec<RadixCiphertext> =
        vec![server_key.create_trivial_radix(0, nb_blocks); mat.len()];
    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                let s = shifts[k * size + j];
                if s > 0 {
                    let tmp = server_key.scalar_left_shift_parallelized(&mat[i * size + k], s);
                    prod[i * size + j] = server_key.add_parallelized(&prod[i * size + j], &tmp);
                } else if s < 0 {
                    let tmp =
                        server_key.scalar_left_shift_parallelized(&mat[i * size + k], s.abs());
                    prod[i * size + j] = server_key.sub_parallelized(&prod[i * size + j], &tmp);
                }
            }
        }
    }
    prod
}

fn extract_submatrix(
    matrix: &[RadixCiphertext],
    height: usize,
    width: usize,
    row: usize,
    col: usize,
    nb_blocks: usize,
    server_key: &ServerKey,
) -> Vec<RadixCiphertext> {
    let mut submatrix: Vec<RadixCiphertext> = Vec::with_capacity(9);
    for i in row..(row + 3) {
        for j in col..(col + 3) {
            if i < height && j < width {
                submatrix.push(matrix[i * width + j].clone());
            } else {
                println!("Out of bounds: ({}, {})", i, j);
                submatrix.push(server_key.create_trivial_radix(0, nb_blocks));
            }
        }
    }
    submatrix
}

fn main() {
    let img = image::open("data/bluehen.png").expect("Failed to open image");
    let img_gray = img.to_luma8();
    let (width, height) = img_gray.dimensions();

    let nb_blocks = 8;
    let start = Instant::now();
    // Generate radix keys
    let (client_key, server_key) = gen_keys_radix(PARAM_MESSAGE_2_CARRY_2_KS_PBS, nb_blocks);
    // Generate key for PBS (without padding)
    let wopbs_key = WopbsKey::new_wopbs_key(
        &client_key,
        &server_key,
        &WOPBS_PARAM_MESSAGE_2_CARRY_2_KS_PBS,
    );
    println!(
        "Key generation done in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    let lut_gen_start = Instant::now();
    println!("Generating LUT.");
    let mut dummy: RadixCiphertext = server_key.create_trivial_radix(2_u64, nb_blocks >> 1);
    dummy = wopbs_key.keyswitch_to_wopbs_params(&server_key, &dummy);
    let mut dummy_blocks = dummy.clone().into_blocks().to_vec();
    for block in &mut dummy_blocks {
        block.degree = Degree::new(3);
    }
    dummy = RadixCiphertext::from_blocks(dummy_blocks);
    let sqrt_lut = wopbs_key.generate_lut_radix(&dummy, |x: u64| x.sqrt());
    println!(
        "LUT generation done in {:?} sec.",
        lut_gen_start.elapsed().as_secs_f64()
    );

    let mut enc_pixels =
        vec![server_key.create_trivial_radix(0, nb_blocks); (height * width) as usize];
    let mut enc_out_pixels =
        vec![server_key.create_trivial_radix(0, nb_blocks); (height * width) as usize];

    // Encrypt image
    for i in 0..(height as usize) {
        for j in 0..(width as usize) {
            let pixel = img_gray.get_pixel(j as u32, i as u32);
            enc_pixels[i * (width as usize) + j] = client_key.encrypt(pixel[0] as u64);
        }
    }

    // Prewitt Mask
    let mx = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
    let my = [-1, -1, -1, 0, 0, 0, 1, 1, 1];

    let start = Instant::now();
    // Perform Prewitt Operator (essentially a convolution)
    enc_out_pixels
        .par_iter_mut()
        .enumerate()
        .for_each(|(index, pixel)| {
            let i = index / (width as usize);
            let j = index % (width as usize);
            if !(i > (height as usize - 3) || (j > (width as usize - 3))) {
                let square = extract_submatrix(
                    &enc_pixels,
                    height as usize,
                    width as usize,
                    i,
                    j,
                    nb_blocks,
                    &server_key,
                );
                let mut gx = square_matmul(&square, &mx, nb_blocks, &server_key);
                let mut gy = square_matmul(&square, &my, nb_blocks, &server_key);
                // Sum each matrix
                for k in 1..9 {
                    gx[0] = server_key.add_parallelized(&gx[0], &gx[k]);
                    gy[0] = server_key.add_parallelized(&gy[0], &gy[k]);
                }

                // Square each sum and add together
                gx[0] = server_key.mul_parallelized(&gx[0], &gx[0]);
                gy[0] = server_key.mul_parallelized(&gy[0], &gy[0]);
                gx[0] = server_key.add_parallelized(&gx[0], &gy[0]);

                // Compute square root with PBS
                // println!("gx[0] degree: {:?}", gx[0].blocks()[0].degree);
                *pixel = ct_lut_eval_quantized_no_gen(
                    gx[0].clone(),
                    nb_blocks,
                    &wopbs_key,
                    &server_key,
                    &sqrt_lut,
                );
            }
        });
    println!(
        "Image generated in {:?} sec.",
        start.elapsed().as_secs_f64()
    );

    // Save the output PNG
    let mut image_buffer = ImageBuffer::<image::Luma<u8>, _>::new(width, height);
    for i in 0..(height as usize) {
        for j in 0..(width as usize) {
            let out_pixel: u64 = client_key.decrypt(&enc_out_pixels[i * (width as usize) + j]);
            image_buffer.put_pixel(i as u32, j as u32, image::Luma([out_pixel as u8]));
        }
    }
    let _ = image::DynamicImage::ImageLuma8(image_buffer).save("out_test.png");
}
