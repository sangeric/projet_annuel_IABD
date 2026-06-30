// src/data/flatten

pub mod image;
pub mod augment;

use crate::tensor::Matrix;
use crate::data::image::LoadedImage;
use crate::training::shuffle::shuffled_indices;
use std::fs;
use std::io::{Read, Write};
use std::fs::File;

pub struct Dataset {
    pub x: Matrix,
    pub labels: Vec<usize>,
    pub class_names: Vec<String>,
}

impl Dataset {
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn num_classes(&self) -> usize {
        self.class_names.len()
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        let mut file = match File::create(path) {
            Ok(f) => f,
            Err(e) => return Err(format!("Failed to create {}: {}", path, e)),
        };

        let n_samples = self.x.rows as u32;
        let n_features = self.x.cols as u32;
        let n_classes = self.class_names.len() as u32;

        // Header
        if let Err(e) = file.write_all(&n_samples.to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }
        if let Err(e) = file.write_all(&n_features.to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }
        if let Err(e) = file.write_all(&n_classes.to_le_bytes()) {
            return Err(format!("write failed: {}", e));
        }

        // Class names — each preceded by its byte length
        for name in &self.class_names {
            let bytes = name.as_bytes();
            let len = bytes.len() as u32;
            if let Err(e) = file.write_all(&len.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
            if let Err(e) = file.write_all(bytes) {
                return Err(format!("write failed: {}", e));
            }
        }

        // Labels
        for &label in &self.labels {
            let l = label as u32;
            if let Err(e) = file.write_all(&l.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
        }

        // Feature matrix — written as a single big block of bytes
        for &value in &self.x.data {
            if let Err(e) = file.write_all(&value.to_le_bytes()) {
                return Err(format!("write failed: {}", e));
            }
        }

        Ok(())
    }

    pub fn load(path: &str) -> Result<Dataset, String> {
        let mut file = match File::open(path) {
            Ok(f) => f,
            Err(e) => return Err(format!("Failed to open {}: {}", path, e)),
        };

        // Small helpers to read fixed-size primitives
        let read_u32 = |file: &mut File| -> Result<u32, String> {
            let mut buf = [0u8; 4];
            if let Err(e) = file.read_exact(&mut buf) {
                return Err(format!("read failed: {}", e));
            }
            Ok(u32::from_le_bytes(buf))
        };
        let read_f32 = |file: &mut File| -> Result<f32, String> {
            let mut buf = [0u8; 4];
            if let Err(e) = file.read_exact(&mut buf) {
                return Err(format!("read failed: {}", e));
            }
            Ok(f32::from_le_bytes(buf))
        };

        // Header
        let n_samples = read_u32(&mut file)? as usize;
        let n_features = read_u32(&mut file)? as usize;
        let n_classes = read_u32(&mut file)? as usize;

        // Class names
        let mut class_names = Vec::new();
        for _ in 0..n_classes {
            let len = read_u32(&mut file)? as usize;
            let mut bytes = vec![0u8; len];
            if let Err(e) = file.read_exact(&mut bytes) {
                return Err(format!("read failed: {}", e));
            }
            match String::from_utf8(bytes) {
                Ok(s) => class_names.push(s),
                Err(e) => return Err(format!("invalid utf8 in class name: {}", e)),
            }
        }

        // Labels
        let mut labels = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            labels.push(read_u32(&mut file)? as usize);
        }

        // Feature data
        let mut x_data = Vec::with_capacity(n_samples * n_features);
        for _ in 0..(n_samples * n_features) {
            x_data.push(read_f32(&mut file)?);
        }

        Ok(Dataset {
            x: Matrix::from_vec(x_data, n_samples, n_features),
            labels,
            class_names,
        })
    }

    // Splits this dataset into a training and test set.
    // Same RNG seed -> same split, for reproducible reports.
    pub fn train_test_split(&self, test_ratio: f32, seed: u64) -> (Dataset, Dataset) {
        let n = self.len();
        let indices = shuffled_indices(n, seed);

        let n_test = (n as f32 * test_ratio) as usize;
        let n_features = self.x.cols;

        let mut train_data = Vec::new();
        let mut train_labels = Vec::new();
        let mut test_data = Vec::new();
        let mut test_labels = Vec::new();

        for (k, &original_index) in indices.iter().enumerate() {
            let mut row = Vec::new();
            for j in 0..n_features {
                row.push(self.x.get(original_index, j));
            }
            let label = self.labels[original_index];

            if k < n_test {
                test_data.extend(row);
                test_labels.push(label);
            } else {
                train_data.extend(row);
                train_labels.push(label);
            }
        }

        let train = Dataset {
            x: Matrix::from_vec(train_data, n - n_test, n_features),
            labels: train_labels,
            class_names: self.class_names.clone(),
        };
        let test = Dataset {
            x: Matrix::from_vec(test_data, n_test, n_features),
            labels: test_labels,
            class_names: self.class_names.clone(),
        };

        (train, test)
    }
}

pub fn load_or_build(
    cache_path: &str,
    root: &str,
    samples_per_class: Option<usize>,
    feature_extractor: fn(&LoadedImage) -> Vec<f32>,
) -> Result<Dataset, String> {
    // Try the cache first
    match Dataset::load(cache_path) {
        Ok(ds) => {
            println!("Loaded dataset from cache: {}", cache_path);
            return Ok(ds);
        }
        Err(_) => {
            println!("No cache at {}, building from images...", cache_path);
        }
    }

    // Cache miss — build from scratch
    let dataset = load_dataset(root, samples_per_class, feature_extractor)?;

    // Try to write the cache. If it fails (e.g. directory doesn't exist),
    // log a warning but still return the dataset — caching is an optimization,
    // not a requirement.
    match dataset.save(cache_path) {
        Ok(_) => println!("Saved dataset to cache: {}", cache_path),
        Err(e) => eprintln!("Warning: failed to save cache: {}", e),
    }

    Ok(dataset)
}

/// Loads an image-classification dataset from a folder structure.
///
/// Every direct subdirectory of `root` is treated as a class.
/// The subfolder's name becomes the class label.
/// Classes are sorted alphabetically so label indices are reproducible.
///
/// Example structure:
///   root/
///     class_a/ *.jpg
///     class_b/ *.jpg
///     class_c/ *.jpg
///
/// Parameters:
/// - `root`: path to the parent folder containing the class subfolders.
/// - `samples_per_class`: optional cap on how many images to load per class.
///                        Use `None` to load all of them.
/// - `feature_extractor`: function that turns a loaded image into a feature vector.
///                        Pass `flatten` for raw pixels or `extract` for engineered
///                        features. Any function with the right signature works —
///                        the library doesn't care what features you choose.
///
/// Returns a Dataset whose `class_names` reflects the folders found.
/// Files that fail to load are skipped with a warning.
pub fn load_dataset(
    root: &str,
    samples_per_class: Option<usize>,
    feature_extractor: fn(&LoadedImage) -> Vec<f32>,
) -> Result<Dataset, String> {
    // ---- Step 1: discover class folders ----
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(e) => return Err(format!("Failed to read root directory {}: {}", root, e)),
    };

    let mut class_names: Vec<String> = Vec::new();
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        
        let file_type = match entry.file_type() {
            Ok(t) => t,
            Err(_) => continue,
        };
        if !file_type.is_dir() {
            continue;
        }
        if let Some(name) = entry.file_name().to_str() {
            class_names.push(name.to_string());
        }
    }

    class_names.sort();

    if class_names.is_empty() {
        return Err(format!("No class subdirectories found in {}", root));
    }

    println!("Discovered {} classes: {:?}", class_names.len(), class_names);

    // ---- Step 2: load every image and apply the feature extractor ----
    let mut x_data = Vec::new();
    let mut labels = Vec::new();

    let mut n_features: Option<usize> = None;

    for (class_index, class_name) in class_names.iter().enumerate() {
        let class_dir = format!("{}/{}", root, class_name);
        println!(
            "Loading class '{}' (label={}) from {}...",
            class_name, class_index, class_dir
        );

        let entries = match fs::read_dir(&class_dir) {
            Ok(entries) => entries,
            Err(e) => {
                eprintln!("  Failed to read directory {}: {}", class_dir, e);
                continue;
            }
        };

        let mut loaded_for_this_class = 0;

        for entry in entries {
            if let Some(limit) = samples_per_class {
                if loaded_for_this_class >= limit {
                    break;
                }
            }

            let entry = match entry {
                Ok(e) => e,
                Err(_) => continue,
            };
            let path = entry.path();
            let path_str = match path.to_str() {
                Some(s) => s,
                None => continue,
            };

            match LoadedImage::load(path_str) {
                Ok(img) => {
                    let features = feature_extractor(&img);

                    // Lock in feature length on first image, then enforce consistency
                    match n_features {
                        None => n_features = Some(features.len()),
                        Some(expected) => {
                            if features.len() != expected {
                                return Err(format!(
                                    "Inconsistent feature length: got {} but expected {} from {}",
                                    features.len(),
                                    expected,
                                    path_str
                                ));
                            }
                        }
                    }

                    x_data.extend(features);
                    labels.push(class_index);
                    loaded_for_this_class += 1;
                }
                Err(e) => {
                    eprintln!("  Skipping {}: {}", path_str, e);
                }
            }
        }

        println!("  Loaded {} images for '{}'", loaded_for_this_class, class_name);
    }

    let n_samples = labels.len();
    let n_features = n_features.unwrap_or(0);
    println!("Total: {} samples, {} features each", n_samples, n_features);

    Ok(Dataset {
        x: Matrix::from_vec(x_data, n_samples, n_features),
        labels,
        class_names,
    })
}

pub fn one_hot_encode(labels: &[usize], n_classes: usize) -> Matrix {
    let mut data = Vec::new();
    for &label in labels {
        for j in 0..n_classes {
            if j == label {
                data.push(1.0_f32);
            } else {
                data.push(-1.0_f32);
            }
        }
    }
    Matrix::from_vec(data, labels.len(), n_classes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dummy_dataset(n_samples: usize, n_features: usize, n_classes: usize) -> Dataset {
        let mut data = Vec::new();
        let mut labels = Vec::new();
        let mut class_names = Vec::new();
        for c in 0..n_classes {
            class_names.push(format!("class_{}", c));
        }
        for i in 0..n_samples {
            for _ in 0..n_features {
                data.push(i as f32);
            }
            labels.push(i % n_classes);
        }
        Dataset {
            x: Matrix::from_vec(data, n_samples, n_features),
            labels,
            class_names,
        }
    }

    #[test]
    fn test_dataset_len() {
        let ds = make_dummy_dataset(10, 5, 3);
        assert_eq!(ds.len(), 10);
    }

    #[test]
    fn test_num_classes() {
        let ds = make_dummy_dataset(10, 5, 4);
        assert_eq!(ds.num_classes(), 4);
    }

    #[test]
    fn test_split_sizes() {
        let ds = make_dummy_dataset(100, 5, 3);
        let (train, test) = ds.train_test_split(0.2, 42);
        assert_eq!(train.len(), 80);
        assert_eq!(test.len(), 20);
    }

    #[test]
    fn test_split_is_reproducible() {
        let ds = make_dummy_dataset(50, 3, 3);
        let (train_a, test_a) = ds.train_test_split(0.2, 42);
        let (train_b, test_b) = ds.train_test_split(0.2, 42);
        assert_eq!(train_a.labels, train_b.labels);
        assert_eq!(test_a.labels, test_b.labels);
    }

    #[test]
    fn test_split_different_seeds_differ() {
        let ds = make_dummy_dataset(50, 3, 3);
        let (_, test_a) = ds.train_test_split(0.2, 42);
        let (_, test_b) = ds.train_test_split(0.2, 7);
        assert_ne!(test_a.labels, test_b.labels);
    }

    #[test]
    fn test_split_preserves_all_samples() {
        let ds = make_dummy_dataset(20, 2, 3);
        let (train, test) = ds.train_test_split(0.3, 42);
        assert_eq!(train.len() + test.len(), 20);
    }

    #[test]
    fn test_split_carries_class_names() {
        let ds = make_dummy_dataset(20, 2, 3);
        let (train, test) = ds.train_test_split(0.3, 42);
        assert_eq!(train.class_names, vec!["class_0", "class_1", "class_2"]);
        assert_eq!(test.class_names, vec!["class_0", "class_1", "class_2"]);
    }

    #[test]
    fn test_one_hot_encode() {
        let labels = vec![0, 2, 1];
        let y = one_hot_encode(&labels, 3);

        assert_eq!(y.rows, 3);
        assert_eq!(y.cols, 3);

        // Row 0 — label 0 → [+1, -1, -1]
        assert_eq!(y.get(0, 0),  1.0);
        assert_eq!(y.get(0, 1), -1.0);
        assert_eq!(y.get(0, 2), -1.0);

        // Row 1 — label 2 → [-1, -1, +1]
        assert_eq!(y.get(1, 0), -1.0);
        assert_eq!(y.get(1, 1), -1.0);
        assert_eq!(y.get(1, 2),  1.0);

        // Row 2 — label 1 → [-1, +1, -1]
        assert_eq!(y.get(2, 0), -1.0);
        assert_eq!(y.get(2, 1),  1.0);
        assert_eq!(y.get(2, 2), -1.0);
    }
}
