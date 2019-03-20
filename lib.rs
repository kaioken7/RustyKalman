extern crate nalgebra;
extern crate rand;

use nalgebra::{
    DMatrix, DVector, new_identity, Inverse, Transpose};

pub struct KalmanFilter {
    // means of world state variables, and the current covariance matrix
    state: DVector<f64>,
    cov: DMatrix<f64>,
    // control + old state -> state and error covariance matrix
    update_trans: Box<Fn(&DVector<f64>, &DVector<f64>) -> DVector<f64>>,
    update_cov: DMatrix<f64>,
    // state to sensor readings, and error covariance matrix
    sensor_trans: Box<Fn(&DVector<f64>) -> DVector<f64>>,
    sensor_cov: DMatrix<f64>
}

pub fn jacobian(vals: &DVector<f64>, f: &Fn(&DVector<f64>) -> DVector<f64>) -> DMatrix<f64> {
    let input_len = vals.len();
    let output_len = f(vals).len();

    let cols: Vec<DVector<f64>> = (0..input_len).map(|i| {
        let mut v = vals.clone();
        let delta: f64 = 1e-8;
        v[i] += delta; // TODO choose a good threshold
        let output1 = f(&v) - f(&vals);
        let output2 = output1.clone() / delta;
        output2
    }).collect();

    DMatrix::from_fn(output_len, input_len, |o, i| cols[i][o])
}

impl KalmanFilter {
    fn update(&mut self, control_data_raw: &Vec<f64>, sensor_data_raw: &Vec<f64>) {
        let control_data = DVector::from_slice(control_data_raw.len(), control_data_raw);
        let sensor_data = DVector::from_slice(sensor_data_raw.len(), sensor_data_raw);

        let update_trans = &self.update_trans;
        let sensor_trans = &self.sensor_trans;
        let g = jacobian(&self.state, &|x| update_trans(&sensor_data, &x)); // TODO FIX
        let h = jacobian(&self.state, &|x| sensor_trans(&x));

        let state_bar = update_trans(&control_data, &self.state);
        let cov_bar = &g * &self.cov * &g.transpose() + &self.update_cov;
        let kalman_gain = &cov_bar *
                          &h.transpose() *
                          (&h * &cov_bar * &h.transpose() + &self.sensor_cov).inverse().unwrap();
        self.state = state_bar.clone() + &kalman_gain * (sensor_data - sensor_trans(&state_bar));
        self.cov = (new_identity::<DMatrix<f64>>(state_bar.len()) - &kalman_gain * &h) * cov_bar;
    }
}

#[cfg(test)]
mod test {
    use jacobian;
    use KalmanFilter;
    use na::{DMatrix, DVector};
    use rand;
    use rand::distributions::{Normal, IndependentSample};
    #[test]
    fn it_works() {
        let mut k = KalmanFilter {
            state: DVector::from_elem(2, 0.0),
            cov: DMatrix::from_row_vector(2, 2, &vec![10000.0, 0.0, 0.0, 10000.0]),
            update_trans: Box::new(|_, x| DVector::from_slice(2, &vec![x[0] + x[1], x[1]])),
            update_cov: DMatrix::from_row_vector(2, 2, &vec![0.0, 0.0, 0.0, 0.0]),
            sensor_trans: Box::new(|x| DVector::from_elem(1, x[0])),
            sensor_cov: DMatrix::from_elem(1, 1, 300.0),
        };
        let true_value = |t| 4.0 * (t as f64) + 100.0;
        println!("time start state {:?}", k.state);
        let mut r = rand::thread_rng();
        let n = Normal::new(0.0, 60.0);
        for t in 0..1000 {
            let noise = n.ind_sample(&mut r);
            let loc_w_noise = noise + true_value(t);
            k.update(&vec![0.0], &vec![loc_w_noise]);
            println!("time {} noise {} incorrectness {} state {:?}", t, noise, true_value(t) - k.state[0], k.state);
        }
        // KalmanFilter::new(&vec![0.0,0.0], &vec![0.0,0.0,0.0,0.0], &vec![0.0,0.0,0.0,0.0], &vec![0.0,0.0,0.0,0.0], &vec![0.0,0.0,0.0,0.0], &vec![0.0,0.0,0.0,0.0]);
        assert!(false);
    }
}
