pub mod function;
pub use function::train_linear;

pub struct Model{
    pub bias: f32,
    pub weight: Vec<f32>,
}