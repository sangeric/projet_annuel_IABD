// src/training/metrics.rs

pub fn confusion_matrix(predictions: &[usize], labels: &[usize], n_classes: usize) -> Vec<Vec<usize>> {
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

pub fn print_confusion_matrix(matrix: &[Vec<usize>], class_names: &[String]) {
    let n = class_names.len();

    let mut col_width = 0;
    for name in class_names {
        if name.len() + 5 > col_width {
            col_width = name.len() + 5;
        }
    }
    
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

    print!("{:>width$}", "", width = label_width);
    for name in class_names {
        print!(" {:>width$}", format!("pred {}", name), width = col_width);
    }
    println!();

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
        let labels = vec![0, 1, 2];
        let predictions = vec![1, 2, 0];
        let matrix = confusion_matrix(&predictions, &labels, 3);

        assert_eq!(matrix[0][0], 0);
        assert_eq!(matrix[1][1], 0);
        assert_eq!(matrix[2][2], 0);

        assert_eq!(matrix[0][1], 1); // true 0, predicted 1
        assert_eq!(matrix[1][2], 1); // true 1, predicted 2
        assert_eq!(matrix[2][0], 1); // true 2, predicted 0
    }

    #[test]
    fn test_row_sums_match_label_counts() {
        let labels = vec![0, 0, 0, 1, 1, 2];
        let predictions = vec![0, 1, 2, 1, 0, 2];
        let matrix = confusion_matrix(&predictions, &labels, 3);

        let mut row_0 = 0;
        for v in &matrix[0] {
            row_0 += v;
        }
        assert_eq!(row_0, 3);

        let mut row_1 = 0;
        for v in &matrix[1] {
            row_1 += v;
        }
        assert_eq!(row_1, 2);

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

        assert_eq!(matrix.len(), 5);
        for row in &matrix {
            assert_eq!(row.len(), 5);
        }
    }
}
