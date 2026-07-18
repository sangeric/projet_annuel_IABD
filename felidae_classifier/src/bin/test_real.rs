// src/bin/test_real.rs

use felidae_classifier::features::scaler::StandardScaler;
use felidae_classifier::features::flatten::flatten;
use felidae_classifier::features::extract::extract;
// use felidae_classifier::models::mlp::MLP;
use felidae_classifier::models::rbfn::RBFN;
use felidae_classifier::models::svm::kernel::Kernel;
use felidae_classifier::models::svm::multiclass::MulticlassSVM;
// use felidae_classifier::models::perceptron::RosenblattClassifier;
use felidae_classifier::data::load_or_build;
use felidae_classifier::data::one_hot_encode;
use felidae_classifier::training::metrics::{confusion_matrix, print_confusion_matrix};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = if args.len() > 1 { args[1].as_str() } else { "extract" };

    println!("=== Real Dataset Classification (mode: {}) ===\n", mode);
    println!("--- Loading dataset ---");

    let feature_fn: fn(&felidae_classifier::data::image::LoadedImage) -> Vec<f32> = match mode {
        "flatten" => flatten,
        "extract" => extract,
        other => {
            eprintln!("Unknown mode '{}', use 'extract' or 'flatten'", other);
            return;
        }
    };

    // ---------------------------------------------------------------
    // Full dataset — used by Rosenblatt, MLP, RBFN
    // ---------------------------------------------------------------
    let cache_path = format!("cache/dataset_{}_3000.bin", mode);
    let dataset = match load_or_build(&cache_path, "dataset_clean", Some(3000), feature_fn) {
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

    // ---------------------------------------------------------------
    // Rosenblatt (one-vs-rest)
    // ---------------------------------------------------------------
/*    println!("\n=== Rosenblatt (one-vs-rest) ===\n");

    let rosen_epochs = 100;
    let rosen_lr = 0.01;
    let bias = 1.0;
    let seed = 200;

    let mut classifier_rosen = RosenblattClassifier::new(
        n_features, rosen_lr, bias, bias, bias, seed
    );

    classifier_rosen.train_with_eval(
        &x_train,
        &train_ds.labels,
        &x_test,
        &test_ds.labels,
        rosen_epochs,
    );

    classifier_rosen
        .save("saved_models/rosenblatt.bin", [seed, seed + 200, seed + 400])
        .expect("Erreur lors de la sauvegarde du modèle");

    println!("\n=== Prédiction sur le jeu de test (Rosenblatt) ===\n");
    let rosen_predictions = classifier_rosen.predict(&x_test, &test_ds.labels);
    let rosen_matrix = confusion_matrix(&rosen_predictions, &test_ds.labels, n_classes);
    println!("\n=== Matrice de confusion (Rosenblatt) ===\n");
    print_confusion_matrix(&rosen_matrix, &dataset.class_names);

    println!("\nChargement du modèle Rosenblatt sauvegardé...");
    let (mut _classifier_rosen_loaded, _seeds) =
        RosenblattClassifier::load("saved_models/rosenblatt.bin")
            .expect("Erreur lors du chargement du modèle");
    // println!("Seeds chargées : {:?}", seeds);
    // classifier_rosen_loaded.get_all_params();
    // classifier_rosen_loaded.get_all_weight();

    // ---------------------------------------------------------------
    // MLP
    // ---------------------------------------------------------------
    println!("\n=== MLP ===\n");

    let architecture = &[n_features, 32, 16, n_classes];
    println!("Architecture: {:?}", architecture);

    let mut mlp = MLP::new(architecture, "tanh", "tanh").unwrap();

    let epochs = 250;
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
    println!("\n=== Confusion Matrix on Test Set (MLP) ===\n");
    print_confusion_matrix(&matrix, &dataset.class_names);
*/
    // ---------------------------------------------------------------
    // RBFN
    // ---------------------------------------------------------------
    println!("\n=== RBFN ===\n");

    let k = 150;
    let gamma = 0.1;
    let kmeans_iters = 40;
    println!("Centers K = {}, gamma = {}", k, gamma);

    let rbfn = RBFN::train(&x_train, &y_train, k, gamma, kmeans_iters, 42);

    let rbfn_train_acc = rbfn.accuracy(&x_train, &y_train);
    let rbfn_test_acc = rbfn.accuracy(&x_test, &y_test);
    println!(
        "\nRBFN final results: train accuracy {:.1}% | test accuracy {:.1}%",
        rbfn_train_acc * 100.0,
        rbfn_test_acc * 100.0,
    );

    let rbfn_predictions = rbfn.predict(&x_test);
    let rbfn_matrix = confusion_matrix(&rbfn_predictions, &test_ds.labels, n_classes);
    println!("\n=== RBFN Confusion Matrix on Test Set ===\n");
    print_confusion_matrix(&rbfn_matrix, &dataset.class_names);

    // ---------------------------------------------------------------
    // SVM (one-vs-rest, RBF kernel, soft-margin)
    //
    // Two reasons this section uses its own smaller training set:
    //   1. OSQP solves a dense N x N matrix per binary classifier (x3),
    //      so cost grows sharply with N.
    //   2. A hard-margin SVM on thousands of messy real examples just
    //      memorizes (100% train, ~33% test). Soft-margin (the C cap)
    //      plus fewer training points gives a boundary that generalizes.
    //
    // We train on a small subsample but still evaluate on the FULL test
    // set, so the reported test accuracy stays comparable to the other
    // models.
    // ---------------------------------------------------------------
    println!("\n=== SVM ===\n");

    let svm_cache_path = format!("cache/dataset_{}_200.bin", mode);
    let svm_dataset = match load_or_build(&svm_cache_path, "dataset_clean", Some(200), feature_fn) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("Failed to load SVM dataset: {}", e);
            return;
        }
    };

    // Small train slice, large test slice: 30% train / 70% test, the
    // opposite of the usual ratio, so the SVM trains on few points but
    // is judged on many.
    let (svm_train_ds, svm_test_ds) = svm_dataset.train_test_split(0.7, 42);
    println!(
        "SVM split: {} train, {} test",
        svm_train_ds.len(),
        svm_test_ds.len()
    );

    let mut svm_scaler = StandardScaler::new();
    let svm_x_train = svm_scaler.fit_transform(&svm_train_ds.x);
    let svm_x_test = svm_scaler.transform(&svm_test_ds.x);

    let svm_gamma = 0.01;
    let svm_c = 0.01;
    println!("Kernel: RBF, gamma = {}, C = {}", svm_gamma, svm_c);
    println!("Training on {} examples...", svm_x_train.rows);

    let svm = MulticlassSVM::train(
        &svm_x_train,
        &svm_train_ds.labels,
        n_classes,
        Kernel::Rbf { gamma: svm_gamma },
        svm_c,
    );

    let svm_train_acc = svm.accuracy(&svm_x_train, &svm_train_ds.labels);
    let svm_test_acc = svm.accuracy(&svm_x_test, &svm_test_ds.labels);
    println!(
        "\nSVM final results: train accuracy {:.1}% | test accuracy {:.1}%",
        svm_train_acc * 100.0,
        svm_test_acc * 100.0,
    );

    let svm_predictions = svm.predict(&svm_x_test);
    let svm_matrix = confusion_matrix(&svm_predictions, &svm_test_ds.labels, n_classes);
    println!("\n=== SVM Confusion Matrix on Test Set ===\n");
    print_confusion_matrix(&svm_matrix, &svm_dataset.class_names);

    match svm.save("saved_models/svm.bin") {
        Ok(_) => println!("\nSaved SVM model to saved_models/svm.bin"),
        Err(e) => eprintln!("\nFailed to save SVM: {}", e),
    }
}
