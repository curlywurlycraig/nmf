use rulinalg::matrix::{Matrix, BaseMatrix};
use rulinalg::vector::Vector;

#[macro_use]
extern crate rulinalg;


/// Holds state of NMF operations. Templates are not updated and assumed to be fixed.
/// This is useful for determining activations from an input against a fixed template library.
///
/// For example, extracting played notes on a guitar.
pub struct FixedTemplateNmf<'a> {
    templates: Matrix<f32>,

    /// Optimisation in Dessien et. al.
    cached_template_part: Matrix<f32>,

    activation_coef: Matrix<f32>,
    input: &'a Matrix<f32>,
    beta: f32
}

impl<'a> FixedTemplateNmf<'a> {
    pub fn new(
        templates: Matrix<f32>,
        activation_coef: Matrix<f32>,
        input: &'a Matrix<f32>,
        beta: f32
    ) -> FixedTemplateNmf<'a> {
        let template_count = templates.cols();

        let e = Matrix::new(1, template_count, vec![1f32; template_count]);

        let v_e_t = input * &e;
        let cached_template_part = templates.elemul(&v_e_t).transpose();

        FixedTemplateNmf {
            templates,
            cached_template_part,
            activation_coef,
            input,
            beta
        }
    }

    pub fn update_activation_coef(&mut self) {
        let input_height = self.input.rows();

        let mut resulting_coef: Matrix<f32> = self.activation_coef.clone();

        let w_h: Matrix<f32> = &self.templates * &resulting_coef;
        let w_h_beta_1: Vector<f32> = w_h
            .iter()
            .map(|x| x.powf(self.beta - 1f32))
            .collect();
        let w_h_beta_1: Matrix<f32> = Matrix::new(input_height, 1, w_h_beta_1);

        let w_h_beta_2: Vector<f32> = w_h
            .iter()
            .map(|x| x.powf(self.beta - 2f32))
            .collect();
        let w_h_beta_2: Matrix<f32> = Matrix::new(input_height, 1, w_h_beta_2);

        let numerator = &self.cached_template_part * &w_h_beta_2;
        let denominator = self.templates.transpose() * &w_h_beta_1;
        let fraction = numerator.elediv(&denominator);

        resulting_coef = resulting_coef.elemul(&fraction);

        self.activation_coef = resulting_coef
    }

    pub fn get_activation_coef(&'a self) -> &'a Matrix<f32> {
        &self.activation_coef
    }
}

#[cfg(test)]
mod compute_activation_matrix_tests {
    use rulinalg::matrix::{Matrix, BaseMatrix, Axes};
    use rulinalg::vector::Vector;

    use crate::FixedTemplateNmf;

    fn max_activation_vector(activations: &Matrix<f32>) -> Vector<f32> {
        activations.max(Axes::Col)
    }

    #[test]
    fn it_works_for_some_reasonable_inputs() {
        let templates: Matrix<f32> = matrix![
            1.0, 2.0, 0.0, 30.0, 1.5;
            0.0, 3.0, 1.0, 30.0, 1.6;
            5.0, 1.0, 10.0, 30.0, 1.7
        ];

        let activation_coef: Matrix<f32> = matrix![1.0; 1.0; 1.0; 1.0; 1.0];

        let input: Matrix<f32> = matrix![
            2.1;
            3.2;
            0.9
        ];

        let mut nmf = FixedTemplateNmf::new(templates, activation_coef, &input, 0.5);

        for _ in 1..5 {
            nmf.update_activation_coef();
        }

        let activation = nmf.get_activation_coef();
        let result = max_activation_vector(&activation);

        // Note that the max here is index 1 (2.0, 3.0, 1.0)
        assert_eq!(result, Vector::new(vec![0.0038612997, 0.113134526, 0.003651987, 0.057484213, 0.054535303]));
    }
}