use rulinalg::matrix::{Matrix, BaseMatrix};
use rulinalg::vector::Vector;

const ITERATIONS: u16 = 5;
const BETA: f32 = 0.5f32;

pub fn compute_activation_matrix(
    templates: &Matrix<f32>,
    activation_coef: &Vector<f32>,
    input_spectrogram: &Matrix<f32>) -> Vec<f32> {
        let template_count = templates.cols();
        let spectrogram_bin_count = input_spectrogram.rows();

        let e = Matrix::new(1, template_count, vec![1f32; template_count]);

        let v_e_t = input_spectrogram * &e;
        let cached_template_part = templates.elemul(&v_e_t).transpose();

        let mut resulting_coef: Matrix<f32> = Matrix::from(activation_coef.clone());

        for _ in 1..ITERATIONS {
            let w_h: Matrix<f32> = templates * &resulting_coef;
            let w_h_beta_1: Vector<f32> = w_h
                .iter()
                .map(|x| x.powf(BETA - 1f32))
                .collect();
            let w_h_beta_1: Matrix<f32> = Matrix::new(spectrogram_bin_count, 1, w_h_beta_1);

            let w_h_beta_2: Vector<f32> = w_h
                .iter()
                .map(|x| x.powf(BETA - 2f32))
                .collect();
            let w_h_beta_2: Matrix<f32> = Matrix::new(spectrogram_bin_count, 1, w_h_beta_2);

            let numerator = &cached_template_part * &w_h_beta_2;
            let denominator = templates.transpose() * &w_h_beta_1;
            let fraction = numerator.elediv(&denominator);

            resulting_coef = resulting_coef.elemul(&fraction);
        }

        resulting_coef.into_vec()
}

#[cfg(test)]
mod compute_activation_matrix_tests {
    use rulinalg::{matrix::Matrix, vector, vector::Vector};
    use crate::compute_activation_matrix;

    fn max_index(activations: &Vec<f32>) -> usize {
        activations
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index).unwrap()
    }

    #[test]
    fn it_works_for_some_reasonable_inputs() {
        let templates: Matrix<f32> = matrix![
            1.0, 2.0, 0.0, 30.0, 1.5;
            0.0, 3.0, 1.0, 30.0, 1.6;
            5.0, 1.0, 10.0, 30.0, 1.7
        ];

        let activation_coef: Vector<f32> = vector![1.0, 1.0, 1.0, 1.0, 1.0];

        // Straight forward case
        let input_spectrogram: Matrix<f32> = matrix![
            2.1;
            3.2;
            0.9
        ];
        let activation = &compute_activation_matrix(&templates, &activation_coef, &input_spectrogram);
        let result = max_index(&activation);
        println!("Result is {:#?}", &activation);
        assert_eq!(result, 1);

        // What if the amplitude is just lower?
        let input_spectrogram: Matrix<f32> = matrix![
            1.6;
            1.6;
            1.6
        ];
        let activation = &compute_activation_matrix(&templates, &activation_coef, &input_spectrogram);
        let result = max_index(&activation);
        println!("Result is {:#?}", &activation);
        assert_eq!(result, 1); // Look! It's not 3!
        // But, it is actually close if you inspect the output.
        // NMF still gives reasonably high activation for the fourth column.

        // Check off-by-one
        let input_spectrogram: Matrix<f32> = matrix![
            1.0;
            0.1;
            5.1
        ];
        let activation = &compute_activation_matrix(&templates, &activation_coef, &input_spectrogram);
        let result = max_index(&activation);
        println!("Result is {:#?}", &activation);
        assert_eq!(result, 0);
    }
}