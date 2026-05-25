pub mod guard;
pub mod memory;
pub mod queue;

use guard::Guard;
use memory::NeuronField;
use queue::{EventPacket, EventQueue};
use std::collections::{HashMap, HashSet};
use std::error::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DBpediaCategory {
    Company = 0,
    EducationalInstitution = 1,
    Artist = 2,
    Athlete = 3,
    OfficeHolder = 4,
    MeanOfTransportation = 5,
    Building = 6,
    NaturalPlace = 7,
    Village = 8,
    Animal = 9,
    Plant = 10,
    Album = 11,
    Film = 12,
    WrittenWork = 13,
}

impl DBpediaCategory {
    pub fn from_class_index(index: u32) -> Self {
        if index < 1 || index > 14 {
            panic!("Invalid class index: {}", index);
        }
        unsafe { std::mem::transmute((index - 1) as u8) }
    }

    pub fn name(&self) -> &'static str {
        match self {
            DBpediaCategory::Company => "Company",
            DBpediaCategory::EducationalInstitution => "EducationalInstitution",
            DBpediaCategory::Artist => "Artist",
            DBpediaCategory::Athlete => "Athlete",
            DBpediaCategory::OfficeHolder => "OfficeHolder",
            DBpediaCategory::MeanOfTransportation => "MeanOfTransportation",
            DBpediaCategory::Building => "Building",
            DBpediaCategory::NaturalPlace => "NaturalPlace",
            DBpediaCategory::Village => "Village",
            DBpediaCategory::Animal => "Animal",
            DBpediaCategory::Plant => "Plant",
            DBpediaCategory::Album => "Album",
            DBpediaCategory::Film => "Film",
            DBpediaCategory::WrittenWork => "WrittenWork",
        }
    }
}

/// Fast-path signal propagation for Run Mode.
pub fn propagate_run(field: &NeuronField, queue: &EventQueue, packet: EventPacket) {
    unsafe {
        let neuron = field.get_neuron(packet.target_id as usize);
        neuron.potential += packet.magnitude;

        if neuron.potential >= neuron.threshold {
            neuron.potential = 0.0; // Reset potential on fire

            if neuron.target_id != 999 && (neuron.target_id as usize) < field.size {
                let next_packet = EventPacket {
                    target_id: neuron.target_id,
                    magnitude: neuron.weight,
                    source_id: None,
                };
                queue.push(next_packet);
            }
        }
    }
}

/// Transactional signal propagation for Trainer Mode.
pub fn propagate_trainer(
    field: &NeuronField,
    neuron_id: u32,
    magnitude: f32,
    parent_guard: Option<&Guard>,
    feedback_value: f32,
) {
    unsafe {
        let neuron = field.get_neuron(neuron_id as usize);
        let current_guard = Guard::new(neuron_id, field, parent_guard);

        let incoming_signal = if parent_guard.is_some() {
            let parent_neuron = field.get_neuron(parent_guard.unwrap().neuron_id as usize);
            magnitude * parent_neuron.weight
        } else {
            magnitude
        };

        neuron.potential += incoming_signal;

        if neuron.potential >= neuron.threshold {
            let target_id = neuron.target_id;

            if target_id != 999 && (target_id as usize) < field.size {
                propagate_trainer(
                    field,
                    target_id,
                    magnitude,
                    Some(&current_guard),
                    feedback_value,
                );
            } else {
                current_guard.propagate_feedback(feedback_value);
            }
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
    println!("📚 DBpedia 560,000 Ontology Dataset Classification PoC 📚");
    println!("====================================================================\n");

    // Define a set of common stop words to exclude from our vocabulary
    let stop_words: HashSet<&str> = [
        "the", "a", "and", "of", "to", "in", "on", "for", "with", "at", "by", "an", "be", "is",
        "are", "was", "were", "it", "that", "this", "from", "as", "at", "but", "not", "or", "will",
        "has", "have", "its", "his", "her", "their", "they", "who", "which", "which", "also",
        "been", "by", "an", "about",
    ]
    .iter()
    .cloned()
    .collect();

    println!("--- Step 1: Building Vocabulary from 560,000 Training Samples ---");
    let train_file_path = "example/dbpedia/dbpedia_csv/train.csv";
    let mut rdr = csv::Reader::from_path(train_file_path)?;

    // Count word frequencies per category to build a discriminative vocabulary
    let mut word_counts: HashMap<String, [u32; 14]> = HashMap::new();

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
                let counts = word_counts.entry(token).or_insert([0; 14]);
                counts[cat_idx as usize] += 1;
            }
        }
    }

    // Sort words by total frequency and keep the top 2,000 most frequent words
    let mut word_list: Vec<(String, [u32; 14], u32)> = word_counts
        .into_iter()
        .map(|(word, counts)| {
            let total_count: u32 = counts.iter().sum();
            (word, counts, total_count)
        })
        .collect();

    word_list.sort_by(|a, b| b.2.cmp(&a.2)); // Sort descending
    let vocab_size = 2000;
    let final_vocab: Vec<(String, [u32; 14])> = word_list
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
    let num_experts = 14;
    let field_size = num_words + num_experts; // 2014 neurons

    println!("Vocabulary built successfully!");
    println!("  Top 2,000 most frequent words selected.");
    println!("  Total Neuron Field Size: {} neurons\n", field_size);

    // 3. Initialize the NeuronField
    let field = NeuronField::new(field_size);

    // Configure word neurons to target their respective experts (2000..2014)
    // Each word targets the expert (category) in which it occurs most frequently!
    unsafe {
        for i in 0..num_words {
            let n = field.get_neuron(i);
            n.potential = 0.0;
            n.threshold = 1.0;

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

            n.target_id = (num_words + max_idx) as u32;
            n.weight = 1.5; // Start with high weight (susceptible to noise)
        }

        // Configure Expert neurons (2000..2014)
        for i in num_words..field_size {
            let n = field.get_neuron(i);
            n.potential = 0.0;
            n.threshold = 1.0;
            n.target_id = 999; // End of chain
            n.weight = 0.0;
        }
    }

    println!("--- Step 2: Training on 560,000 Samples (Trainer Mode) ---");
    println!("Applying the Guard feedback loop over the entire dataset...");

    let start_time = std::time::Instant::now();
    let mut rdr = csv::Reader::from_path(train_file_path)?;
    let mut sample_count = 0;

    for result in rdr.records() {
        let record = result?;
        let class_index: u32 = record[0].parse()?;
        let category = DBpediaCategory::from_class_index(class_index);
        let title = &record[1];
        let description = &record[2];

        let full_text = format!("{} {}", title, description);
        let tokens = tokenize(&full_text);

        for token in tokens {
            if let Some(&word_idx) = vocab_map.get(&token) {
                unsafe {
                    let word_neuron = field.get_neuron(word_idx);
                    let target_expert = word_neuron.target_id - num_words as u32;

                    // Positive feedback if correct expert, negative if incorrect
                    let feedback = if target_expert == category as u32 {
                        0.05
                    } else {
                        -0.15
                    };

                    propagate_trainer(&field, word_idx as u32, 1.0, None, feedback);
                }
            }
        }

        sample_count += 1;
        if sample_count % 100000 == 0 {
            println!("  Processed {}/560,000 samples...", sample_count);
        }
    }

    let duration = start_time.elapsed();
    println!("Training completed in {:.2?}!", duration);

    println!("\n--- Step 3: Evaluating on 70,000 Test Samples (Run Mode) ---");
    let test_file_path = "example/dbpedia/dbpedia_csv/test.csv";
    let mut rdr = csv::Reader::from_path(test_file_path)?;

    let mut correct_predictions = 0;
    let mut total_predictions = 0;

    // Confusion matrix: [Actual][Predicted]
    let mut confusion_matrix = vec![vec![0u32; 14]; 14];

    for result in rdr.records() {
        let record = result?;
        let class_index: u32 = record[0].parse()?;
        let actual_category = DBpediaCategory::from_class_index(class_index);
        let title = &record[1];
        let description = &record[2];

        let full_text = format!("{} {}", title, description);
        let tokens = tokenize(&full_text);

        // Reset expert potentials
        unsafe {
            for i in num_words..field_size {
                field.get_neuron(i).potential = 0.0;
            }
        }

        // Present each word in Run Mode
        for token in tokens {
            if let Some(&word_idx) = vocab_map.get(&token) {
                unsafe {
                    let word_neuron = field.get_neuron(word_idx);
                    let target_expert = word_neuron.target_id;
                    let expert_neuron = field.get_neuron(target_expert as usize);
                    expert_neuron.potential += word_neuron.weight;
                }
            }
        }

        // Determine which expert has the highest potential
        let mut predicted_idx = 0;
        let mut max_potential = -1.0;
        unsafe {
            for idx in 0..14 {
                let p = field.get_neuron(num_words + idx).potential;
                if p > max_potential {
                    max_potential = p;
                    predicted_idx = idx;
                }
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
    print!("  Actual \\ Predicted");
    for i in 0..14 {
        print!(" | C{:02}", i + 1);
    }
    println!();
    println!("  -------------------|----|----|----|----|----|----|----|----|----|----|----|----|----|----");
    for i in 0..14 {
        let cat = DBpediaCategory::from_class_index((i + 1) as u32);
        let mut cat_name = cat.name().to_string();
        if cat_name.len() > 17 {
            cat_name.truncate(17);
        }
        print!("  {:17} |", cat_name);
        for j in 0..14 {
            print!(" {:2} |", confusion_matrix[i][j]);
        }
        println!();
    }
    println!("\nLegend:");
    for i in 0..14 {
        let cat = DBpediaCategory::from_class_index((i + 1) as u32);
        println!("  C{:02}: {}", i + 1, cat.name());
    }
    println!("====================================================================");

    Ok(())
}
