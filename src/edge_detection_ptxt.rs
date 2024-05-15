use image::ImageBuffer;
use num_integer::Roots;

fn square_matmul(mat: &[u64], shifts: &[i32]) -> Vec<u64> {
    let size = mat.len().sqrt();
    let mut prod: Vec<u64> = vec![0; mat.len()];
    for i in 0..size {
        for j in 0..size {
            for k in 0..size {
                let s = shifts[k * size + j];
                if s > 0 {
                    prod[i * size + j] += mat[i * size + k] << s;
                } else if s < 0 {
                    prod[i * size + j] -= mat[i * size + k] << s.abs();
                }
            }
        }
    }
    prod
}

fn extract_submatrix(
    matrix: &[u64],
    height: usize,
    width: usize,
    row: usize,
    col: usize,
) -> Vec<u64> {
    let mut submatrix = Vec::with_capacity(9);
    for i in row..(row + 3) {
        for j in col..(col + 3) {
            if i < height && j < width {
                submatrix.push(matrix[i * width + j]);
            } else {
                println!("Out of bounds: ({}, {})", i, j);
                submatrix.push(0);
            }
        }
    }
    submatrix
}

fn main() {
    let img = image::open("data/house.png").expect("Failed to open image");
    let img_gray = img.to_luma8();
    let (width, height) = img_gray.dimensions();
    let mut pixels: Vec<u64> = vec![0; (width * height) as usize];
    let mut out_pixels: Vec<u64> = vec![0; (width * height) as usize];

    for i in 0..(height as usize) {
        for j in 0..(width as usize) {
            let pixel = img_gray.get_pixel(j as u32, i as u32);
            pixels[i * (width as usize) + j] = pixel[0] as u64;
        }
    }

    // Prewitt Mask
    let mx = [-1, 0, 1, -1, 0, 1, -1, 0, 1];
    let my = [-1, -1, -1, 0, 0, 0, 1, 1, 1];

    // Perform Prewitt Operator (essentially a convolution)
    for i in 0..((height - 2) as usize) {
        for j in 0..((width - 2) as usize) {
            // Gradient approximations
            let square = extract_submatrix(&pixels, height as usize, width as usize, i, j);
            let gx = square_matmul(&square, &mx);
            let gx: u64 = gx.iter().sum();

            let gy = square_matmul(&square, &my);
            let gy: u64 = gy.iter().sum();

            // Calculate magnitude
            let magnitude = (gx.pow(2) + gy.pow(2)).sqrt();

            out_pixels[i * (width as usize) + j] = magnitude;
        }
    }

    // Save the output PNG
    let mut image_buffer = ImageBuffer::<image::Luma<u8>, _>::new(width, height);
    for i in 0..(height as usize) {
        for j in 0..(width as usize) {
            image_buffer.put_pixel(
                j as u32,
                i as u32,
                image::Luma([out_pixels[i * (width as usize) + j] as u8]),
            );
        }
    }
    let _ = image::DynamicImage::ImageLuma8(image_buffer).save("out_test.png");
}
