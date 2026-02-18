/// Operator to reduce a vector to a single value.
#[derive(Debug, PartialEq, Eq, Clone)]
pub enum ElementOperator {
    /// Maximum.
    Max,
    /// Minimum.
    Min,
}

impl ElementOperator {
    /// Returns the evaluation result.
    ///
    /// # Panics
    ///
    /// Panics if a min/max reduce operation is performed on an empty vector.
    pub fn eval<T: PartialOrd + Copy>(&self, vector: &[T]) -> T {
        self.eval_iter(vector.iter().copied()).unwrap()
    }

    /// Returns the evaluation result.
    pub fn eval_iter<T: PartialOrd + Copy , I: Iterator<Item = T>>(
        &self,
        iter: I,
    ) -> Option<T> {
        match self {
            Self::Max => iter.reduce(|x, y| if y > x { y } else { x }),
            Self::Min => iter.reduce(|x, y| if y < x { y } else { x }),
        }
    }
}