// src/training/metrics.rs

/// Builds a confusion matrix from predictions and ground truth labels.
///
/// Returns a 2D matrix where `matrix[true][predicted]` is the count of samples
/// that had `true` as their actual class and were predicted as `predicted`.
///
/// The diagonal contains correct predictions; off-diagonal cells show
/// which classes get confused with which others.
///
/// # Example
/// 3 classes (cat, lion, cheetah):
///
///                pred cat  pred lion  pred cheetah
///   true cat        45         3          2
///   true lion        4        38          8
///   true cheetah     2        12         36
///
/// Reading: of 50 actual cats, 45 were predicted correctly, 3 were called
/// lions, and 2 were called cheetahs.
pub fn confusion_matrix(
    predictions: &[usize],
    labels: &[usize],
    n_classes: usize,
) -> Vec<Vec<usize>> {
    assert_eq!(
        predictions.len(),
        labels.len(),
        "predictions and labels must have the same length"
    );

    // Initialize an n_classes × n_classes matrix of zeros
    let mut matrix = Vec::new();
    for _ in 0..n_classes {
        matrix.push(vec![0_usize; n_classes]);
    }

    // For each sample, increment matrix[true_label][predicted_label]
    for i in 0..predictions.len() {
        let true_label = labels[i];
        let predicted_label = predictions[i];

        // Defensive — skip any labels outside the expected range rather than crash
        if true_label < n_classes && predicted_label < n_classes {
            matrix[true_label][predicted_label] += 1;
        }
    }

    matrix
}

/// Pretty-prints a confusion matrix with class names as labels.
///
/// The output is aligned for readability:
///
/// === Confusion Matrix ===
///                pred cat   pred lion   pred cheetah
///   true cat         45          3            2
///   true lion         4         38            8
///   true cheetah      2         12           36
pub fn print_confusion_matrix(matrix: &[Vec<usize>], class_names: &[String]) {
    let n = class_names.len();

    // Find the widest name so we can align cells
    let mut col_width = 0;
    for name in class_names {
        if name.len() + 5 > col_width {
            // +5 to accommodate the "pred " prefix
            col_width = name.len() + 5;
        }
    }
    // Make sure cells are at least wide enough for the largest count
    let mut max_count = 0;
    for row in matrix {
        for &count in row {
            if count > max_count {
                max_count = count;
            }
        }
    }
    let count_width = max_count.to_string().len();
    if count_width + 2 > col_width {
        col_width = count_width + 2;
    }

    let label_width = class_names.iter().map(|s| s.len()).max().unwrap_or(0) + 5;

    // Header row
    print!("{:>width$}", "", width = label_width);
    for name in class_names {
        print!(" {:>width$}", format!("pred {}", name), width = col_width);
    }
    println!();

    // Data rows
    for i in 0..n {
        print!("{:>width$}", format!("true {}", class_names[i]), width = label_width);
        for j in 0..n {
            print!(" {:>width$}", matrix[i][j], width = col_width);
        }
        println!();
    }
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_predictions() {
        // If predictions match labels, only the diagonal should be non-zero
        let labels = vec![0, 1, 2, 0, 1, 2];
        let predictions = vec![0, 1, 2, 0, 1, 2];
        let matrix = confusion_matrix(&predictions, &labels, 3);

        assert_eq!(matrix[0][0], 2);
        assert_eq!(matrix[1][1], 2);
        assert_eq!(matrix[2][2], 2);

        // Off-diagonal cells should all be zero
        for i in 0..3 {
            for j in 0..3 {
                if i != j {
                    assert_eq!(matrix[i][j], 0);
                }
            }
        }
    }

    #[test]
    fn test_all_wrong_predictions() {
        // Predictions never match — diagonal must all be zero
        let labels = vec![0, 1, 2];
        let predictions = vec![1, 2, 0];
        let matrix = confusion_matrix(&predictions, &labels, 3);

        // Diagonal is zero
        assert_eq!(matrix[0][0], 0);
        assert_eq!(matrix[1][1], 0);
        assert_eq!(matrix[2][2], 0);

        // Specific confusions
        assert_eq!(matrix[0][1], 1); // true 0, predicted 1
        assert_eq!(matrix[1][2], 1); // true 1, predicted 2
        assert_eq!(matrix[2][0], 1); // true 2, predicted 0
    }

    #[test]
    fn test_row_sums_match_label_counts() {
        // Each row's sum equals how many times that class appears in labels
        let labels = vec![0, 0, 0, 1, 1, 2];
        let predictions = vec![0, 1, 2, 1, 0, 2];
        let matrix = confusion_matrix(&predictions, &labels, 3);

        // 3 samples have label 0 → row 0 must sum to 3
        let mut row_0 = 0;
        for v in &matrix[0] {
            row_0 += v;
        }
        assert_eq!(row_0, 3);

        // 2 samples have label 1 → row 1 must sum to 2
        let mut row_1 = 0;
        for v in &matrix[1] {
            row_1 += v;
        }
        assert_eq!(row_1, 2);

        // 1 sample has label 2 → row 2 must sum to 1
        let mut row_2 = 0;
        for v in &matrix[2] {
            row_2 += v;
        }
        assert_eq!(row_2, 1);
    }

    #[test]
    fn test_dimensions() {
        let labels = vec![0, 1, 2, 3, 4];
        let predictions = vec![0, 1, 2, 3, 4];
        let matrix = confusion_matrix(&predictions, &labels, 5);

        // 5 × 5 matrix
        assert_eq!(matrix.len(), 5);
        for row in &matrix {
            assert_eq!(row.len(), 5);
        }
    }
}
