// Copyright 2026 Adam Lusted
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use neuron_poc::neuron_guard::ThreadBoundedNeuronField;
use std::collections::{HashMap, HashSet};
use std::error::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AGNewsCategory {
    World = 0,
    Sports = 1,
    Business = 2,
    SciTech = 3,
}

impl AGNewsCategory {
    pub fn from_class_index(index: u32) -> Self {
        match index {
            1 => AGNewsCategory::World,
            2 => AGNewsCategory::Sports,
            3 => AGNewsCategory::Business,
            4 => AGNewsCategory::SciTech,
            _ => panic!("Invalid class index: {}", index),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            AGNewsCategory::World => "World News",
            AGNewsCategory::Sports => "Sports",
            AGNewsCategory::Business => "Business",
            AGNewsCategory::SciTech => "Sci/Tech",
        }
    }
}

/// Simple text tokenizer and cleaner
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("====================================================================");
    println!("📰 AG News 120,000 Dataset Classification PoC 📰");
    println!("====================================================================\n");

    // Define a set of common stop words to exclude from our vocabulary
    let stop_words: HashSet<&str> = [
        "the", "a", "and", "of", "to", "in", "on", "for", "with", "at", "by", "an", "be", "is",
        "are", "was", "were", "it", "that", "this", "from", "as", "at", "but", "not", "or", "will",
        "has", "have",
    ]
    .iter()
    .cloned()
    .collect();

    println!("--- Step 1: Building Vocabulary from 120,000 Training Samples ---");
    let train_file_path = "example/ag_news/data/train.csv";
    let mut rdr = csv::Reader::from_path(train_file_path)?;

    // Count word frequencies per category to build a discriminative vocabulary
    let mut word_counts: HashMap<String, [u32; 4]> = HashMap::new();

    for result in rdr.records() {
        let record = result?;
        let class_index: u32 = record[0].parse()?;
        let cat_idx = class_index - 1; // 0-based category index
        let title = &record[1];
        let description = &record[2];

        let full_text = format!("{} {}", title, description);
        let tokens = tokenize(&full_text);

        for token in tokens {
            if token.len() > 2 && !stop_words.contains(token.as_str()) {
                let counts = word_counts.entry(token).or_insert([0; 4]);
                counts[cat_idx as usize] += 1;
            }
        }
    }

    // Sort words by total frequency and keep the top 1,000 most frequent words
    let mut word_list: Vec<(String, [u32; 4], u32)> = word_counts
        .into_iter()
        .map(|(word, counts)| {
            let total_count: u32 = counts.iter().sum();
            (word, counts, total_count)
        })
        .collect();

    word_list.sort_by(|a, b| b.2.cmp(&a.2)); // Sort descending
    let vocab_size = 1000;
    let final_vocab: Vec<(String, [u32; 4])> = word_list
        .into_iter()
        .take(vocab_size)
        .map(|(word, counts, _)| (word, counts))
        .collect();

    // Map words to their index in the vocabulary
    let vocab_map: HashMap<String, usize> = final_vocab
        .iter()
        .enumerate()
        .map(|(idx, (word, _))| (word.clone(), idx))
        .collect();

    let num_words = final_vocab.len();
    let num_experts = 4;
    let field_size = num_words + num_experts; // 1004 neurons

    println!("Vocabulary built successfully!");
    println!("  Top 1,000 most frequent words selected.");
    println!(
        "  Total Neuron Field Size: {} neurons (64 bytes each)\n",
        field_size
    );

    // 3. Initialize the ThreadBoundedNeuronField
    let field = ThreadBoundedNeuronField::new(field_size);

    // Configure word neurons to target their respective experts (1000..1004)
    // Each word targets the expert (category) in which it occurs most frequently!
    unsafe {
        for i in 0..num_words {
            let n = field.get_neuron(i);
            n.token_id = i as u32;

            // Find the category with the highest frequency for this word
            let counts = final_vocab[i].1;
            let mut max_idx = 0;
            let mut max_val = 0;
            for (idx, &val) in counts.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    max_idx = idx;
                }
            }

            let target_expert = (num_words + max_idx) as u32;
            n.update_or_add_connection(target_expert, 15); // Start with high weight modifier
        }

        // Configure Expert neurons (1000..1004)
        for i in num_words..field_size {
            let n = field.get_neuron(i);
            n.token_id = i as u32;
            n.active_connections = 0;
        }
    }

    println!("--- Step 2: Training on 120,000 Samples (Trainer Mode) ---");
    println!("Applying the Guard feedback loop over the entire dataset...");

    let start_time = std::time::Instant::now();
    let mut rdr = csv::Reader::from_path(train_file_path)?;
    let mut sample_count = 0;

    for result in rdr.records() {
        let record = result?;
        let class_index: u32 = record[0].parse()?;
        let category = AGNewsCategory::from_class_index(class_index);
        let title = &record[1];
        let description = &record[2];

        let full_text = format!("{} {}", title, description);
        let tokens = tokenize(&full_text);

        for token in tokens {
            if let Some(&word_idx) = vocab_map.get(&token) {
                if let Some(lease) = field.try_acquire_lease(word_idx) {
                    let neuron = lease.neuron();
                    let correct_expert = (num_words + category as usize) as u32;

                    // Amplify correct expert pathway
                    neuron.update_or_add_connection(correct_expert, 5);

                    // Suppress incorrect expert pathways
                    for j in 0..neuron.active_connections as usize {
                        let target = neuron.target_neuron_ids[j];
                        if target != correct_expert && target >= num_words as u32 {
                            neuron.weight_modifiers[j] =
                                neuron.weight_modifiers[j].saturating_sub(15);
                        }
                    }
                }
            }
        }

        sample_count += 1;
        if sample_count % 30000 == 0 {
            println!("  Processed {}/120,000 samples...", sample_count);
        }
    }

    let duration = start_time.elapsed();
    println!("Training completed in {:.2?}!", duration);

    println!("\n--- Step 3: Evaluating on 7,600 Test Samples (Run Mode) ---");
    let test_file_path = "example/ag_news/data/test.csv";
    let mut rdr = csv::Reader::from_path(test_file_path)?;

    let mut correct_predictions = 0;
    let mut total_predictions = 0;

    // Confusion matrix: [Actual][Predicted]
    let mut confusion_matrix = [[0u32; 4]; 4];

    for result in rdr.records() {
        let record = result?;
        let class_index: u32 = record[0].parse()?;
        let actual_category = AGNewsCategory::from_class_index(class_index);
        let title = &record[1];
        let description = &record[2];

        let full_text = format!("{} {}", title, description);
        let tokens = tokenize(&full_text);

        // Reset expert potentials
        let mut expert_potentials = [0i32; 4];

        // Present each word in Run Mode (direct matrix-free evaluation)
        for token in tokens {
            if let Some(&word_idx) = vocab_map.get(&token) {
                unsafe {
                    let n = field.get_neuron(word_idx);
                    for i in 0..n.active_connections as usize {
                        let target = n.target_neuron_ids[i] as usize;
                        if target >= num_words && target < field_size {
                            let expert_idx = target - num_words;
                            expert_potentials[expert_idx] += n.weight_modifiers[i] as i32;
                        }
                    }
                }
            }
        }

        // Determine which expert has the highest potential
        let mut predicted_idx = 0;
        let mut max_potential = i32::MIN;
        for idx in 0..4 {
            let p = expert_potentials[idx];
            if p > max_potential {
                max_potential = p;
                predicted_idx = idx;
            }
        }

        let actual_idx = actual_category as usize;
        confusion_matrix[actual_idx][predicted_idx] += 1;

        if predicted_idx == actual_idx {
            correct_predictions += 1;
        }
        total_predictions += 1;
    }

    let accuracy = (correct_predictions as f32 / total_predictions as f32) * 100.0;
    println!("Evaluation Complete!");
    println!(
        "  Accuracy: {:.2}% ({}/{})",
        accuracy, correct_predictions, total_predictions
    );

    println!("\n--- Confusion Matrix ---");
    println!("  Actual \\ Predicted | World | Sports | Business | Sci/Tech");
    println!("  -------------------|-------|--------|----------|---------");
    for i in 0..4 {
        let cat_name = match i {
            0 => "  World News        ",
            1 => "  Sports            ",
            2 => "  Business          ",
            3 => "  Sci/Tech          ",
            _ => unreachable!(),
        };
        println!(
            "{} | {:5} | {:6} | {:8} | {:8}",
            cat_name,
            confusion_matrix[i][0],
            confusion_matrix[i][1],
            confusion_matrix[i][2],
            confusion_matrix[i][3]
        );
    }
    println!("====================================================================");

    Ok(())
}
