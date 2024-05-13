use image::ImageBuffer;
use ndarray::{s, Array2};

fn main() {
    let img = image::open("data/house.png").expect("Failed to open image");
    let img_gray = img.to_luma8();
    let (width, height) = img_gray.dimensions();
    let mut pixels = Array2::<f32>::from_elem((width as usize, height as usize), 0.0);
    let mut out_pixels = Array2::<f32>::from_elem((width as usize, height as usize), 0.0);

    for y in 0..(width as usize) {
        for x in 0..(height as usize) {
            let pixel = img_gray.get_pixel(x as u32, y as u32);
            let float_value = pixel[0] as f32;
            pixels[[y, x]] = float_value;
        }
    }

    // Prewitt Mask
    let mx =
        Array2::<f32>::from_shape_vec((3, 3), vec![-2.0, 0.0, 2.0, -2.0, 0.0, 2.0, -2.0, 0.0, 2.0])
            .unwrap();
    let my =
        Array2::<f32>::from_shape_vec((3, 3), vec![-2.0, -2.0, -2.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0])
            .unwrap();

    // Perform Prewitt Operator (essentially a convolution)
    for i in 0..((height - 2) as usize) {
        for j in 0..((width - 2) as usize) {
            // Gradient approximations
            let gx = (mx.clone() * pixels.slice(s![i..(i + 3), j..(j + 3)])).sum();
            let gy = (my.clone() * pixels.slice(s![i..(i + 3), j..(j + 3)])).sum();
            // Calculate magnitude
            let magnitude = (gx.powi(2) + gy.powi(2)).sqrt();
            out_pixels[[i + 1, j + 1]] = magnitude;
        }
    }

    // Save the output PNG
    let mut image_buffer = ImageBuffer::<image::Luma<u8>, _>::new(
        out_pixels.ncols() as u32,
        out_pixels.nrows() as u32,
    );
    for y in 0..out_pixels.nrows() {
        for x in 0..out_pixels.ncols() {
            image_buffer.put_pixel(x as u32, y as u32, image::Luma([out_pixels[[y, x]] as u8]));
        }
    }
    let _ = image::DynamicImage::ImageLuma8(image_buffer).save("out_test.png");
}
