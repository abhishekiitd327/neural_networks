use rand::Rng;

pub struct Tensor2D {
    pub rows: usize,
    pub cols: usize,
    data: Box<[f32]>,
}

impl Tensor2D {
    // Creates a new 2D Tensor with values initialized to 0
    pub fn new(rows: usize, cols: usize) -> Tensor2D {
        Tensor2D {
            rows: rows,
            cols: cols,
            data: vec![0.0; rows * cols].into_boxed_slice(),
        }
    }

    // Creates a new 2D Tensor with random values
    pub fn new_rand(rows: usize, cols: usize) -> Tensor2D {
        let mut m = Tensor2D::new(rows, cols);
        let mut _rng = rand::thread_rng();
        for r in 0..rows {
            for c in 0..cols {
                m[r][c] = _rng.gen::<f32>()*2.0 - 1.0;
            }
        }
        m
    }

    pub fn hadamard(t1: &Tensor2D, t2: &Tensor2D) -> Tensor2D {
        if t1.rows != t2.rows || t1.cols != t2.cols {
            panic!(
                "Error: Can't do element-wise multiplication of 2D Tensors with sizes {}x{} and {}x{}",
                t1.rows, t1.cols, t2.rows, t2.cols
            );
        }
        let mut m = Tensor2D::new(t1.rows, t1.cols);
        for r in 0..m.rows {
            for c in 0..m.cols {
                m[r][c] = t1[r][c] * t2[r][c];
            }
        }
        m
    }

    pub fn transpose(&self) -> Tensor2D {
        let mut m = Tensor2D::new(self.cols, self.rows);
        for r in 0..m.rows {
            for c in 0..m.cols {
                m[r][c] = self[c][r];
            }
        }
        m
    }

    pub fn apply_function(&self, f: fn(f32) -> f32) -> Tensor2D {
        let mut m = Tensor2D::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                m[r][c] = f(self[r][c]);
            }
        }
        m
    }

    pub fn index_of_max(&self) -> usize {
        let mut max_value: f32 = f32::NEG_INFINITY;
        let mut max_index: usize = 0;
        for i in 0..self.rows * self.cols {
            if self.data[i] > max_value {
                max_value = self.data[i];
                max_index = i;
            }
        }
        max_index
    }
}

impl std::ops::Index<usize> for Tensor2D {
    type Output = [f32];
    fn index(&self, row: usize) -> &Self::Output {
        &self.data[row * self.cols..(row + 1) * self.cols]
    }
}

impl std::ops::IndexMut<usize> for Tensor2D {
    fn index_mut(&mut self, row: usize) -> &mut Self::Output {
        &mut self.data[row * self.cols..(row + 1) * self.cols]
    }
}

impl std::ops::Mul<f32> for Tensor2D {
    type Output = Tensor2D;
    fn mul(self, other: f32) -> Tensor2D {
        let mut result = Tensor2D::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result[r][c] += self[r][c] * other;
            }
        }
        result
    }
}

impl std::ops::Mul<&Tensor2D> for &Tensor2D {
    type Output = Tensor2D;
    fn mul(self, other: &Tensor2D) -> Tensor2D {
        if self.cols != other.rows {
            panic!(
                "Error: Can't multiply 2D tensors with sizes {}x{} and {}x{}!",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let mut result = Tensor2D::new(self.rows, other.cols);
        for r in 0..self.rows {
            for c in 0..other.cols {
                for i in 0..self.cols {
                    result[r][c] += self[r][i] * other[i][c];
                }
            }
        }
        result
    }
}

impl std::ops::Add<&Tensor2D> for &Tensor2D {
    type Output = Tensor2D;
    fn add(self, other: &Tensor2D) -> Tensor2D {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Error: Cant add 2D tensors with sizes {}x{} and {}x{}!",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let mut result = Tensor2D::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result[r][c] = self[r][c] + other[r][c];
            }
        }
        result
    }
}

impl std::ops::AddAssign<Tensor2D> for Tensor2D {
    fn add_assign(&mut self, other: Tensor2D) {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Error: Cant add 2D tensors with sizes {}x{} and {}x{}!",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        for r in 0..self.rows {
            for c in 0..self.cols {
                self[r][c] += other[r][c];
            }
        }
    }
}

impl std::ops::Sub<&Tensor2D> for &Tensor2D {
    type Output = Tensor2D;
    fn sub(self, other: &Tensor2D) -> Tensor2D {
        if self.rows != other.rows || self.cols != other.cols {
            panic!(
                "Error: Can't add 2D tensors with sizes {}x{} and {}x{}!",
                self.rows, self.cols, other.rows, other.cols
            );
        }
        let mut result = Tensor2D::new(self.rows, self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                result[r][c] = self[r][c] - other[r][c];
            }
        }
        result
    }
}

impl std::fmt::Display for Tensor2D {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut s = String::new();
        for r in 0..self.rows {
            s.push_str("[");
            for c in 0..self.cols {
                s.push_str(&self[r][c].to_string());
                s.push_str(" ");
            }
            s.pop();
            s.push_str("]\n");
        }
        write!(f, "{}", s)
    }
}
