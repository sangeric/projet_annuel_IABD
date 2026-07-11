// src/bin/test_real.rs

// use felidae_classifier::data::load_dataset;
use felidae_classifier::features::scaler::StandardScaler;
//use felidae_classifier::features::flatten::flatten;
// To use features instead, swap the line above for the following line :
use felidae_classifier::features::extract::extract;
use felidae_classifier::models::mlp::MLP;
use felidae_classifier::data::load_or_build;
use felidae_classifier::data::one_hot_encode;
use felidae_classifier::training::metrics::{confusion_matrix, print_confusion_matrix};

fn main() {
    println!("=== Real Dataset Classification ===\n");

    println!("--- Loading dataset ---");
    let dataset = match load_or_build("cache/dataset_extract_3000.bin", "dataset_clean", Some(3000), extract) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("Failed to load dataset: {}", e);
            return;
        }
    };

    let n_features = dataset.x.cols;
    let n_classes = dataset.num_classes();

    println!(
        "\nDataset loaded: {} samples, {} features, {} classes",
        dataset.len(),
        n_features,
        n_classes,
    );
    println!("Classes: {:?}", dataset.class_names);

    let (train_ds, test_ds) = dataset.train_test_split(0.2, 42);
    println!("\nSplit: {} train, {} test", train_ds.len(), test_ds.len());

    let mut scaler = StandardScaler::new();
    let x_train = scaler.fit_transform(&train_ds.x);
    let x_test = scaler.transform(&test_ds.x);
    let y_train = one_hot_encode(&train_ds.labels, n_classes);
    let y_test = one_hot_encode(&test_ds.labels, n_classes);

    println!("\n=== MLP ===\n");

    let architecture = &[n_features, 32, 16, n_classes];
    println!("Architecture: {:?}", architecture);

    let mut mlp = MLP::new(architecture, "tanh", "tanh").unwrap();

    let epochs = 100;
    let learning_rate = 0.01;
    println!("Training {} epochs at learning rate {}\n", epochs, learning_rate);
    mlp.train(&x_train, &y_train, &x_test, &y_test, epochs, learning_rate, "./runs/logdir");
    
    let train_acc = mlp.accuracy(&x_train, &y_train);
    let test_acc = mlp.accuracy(&x_test, &y_test);
    println!(
        "\nMLP final results: train accuracy {:.1}% | test accuracy {:.1}%",
        train_acc * 100.0,
        test_acc * 100.0,
    );

    let predictions = mlp.predict(&x_test);
    let matrix = confusion_matrix(&predictions, &test_ds.labels, n_classes);

    println!("\n=== Confusion Matrix on Test Set ===\n");
    print_confusion_matrix(&matrix, &dataset.class_names);
}
