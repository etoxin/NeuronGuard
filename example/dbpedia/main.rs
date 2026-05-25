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

use neuron_poc::guard::Guard;
use neuron_poc::memory::NeuronField;
use neuron_poc::queue::{EventPacket, EventQueue};
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

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
            DBpediaCategory::EducationalInstitution => "Educational Institution",
            DBpediaCategory::Artist => "Artist",
            DBpediaCategory::Athlete => "Athlete",
            DBpediaCategory::OfficeHolder => "Office Holder",
            DBpediaCategory::MeanOfTransportation => "Mean of Transportation",
            DBpediaCategory::Building => "Building",
            DBpediaCategory::NaturalPlace => "Natural Place",
            DBpediaCategory::Village => "Village",
            DBpediaCategory::Animal => "Animal",
            DBpediaCategory::Plant => "Plant",
            DBpediaCategory::Album => "Album",
            DBpediaCategory::Film => "Film",
            DBpediaCategory::WrittenWork => "Written Work",
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
    println!("📚 DBpedia Ontology Interactive CLI Tool 📚");
    println!("====================================================================\n");

    // Define a set of common stop words to exclude from our vocabulary
    let stop_words: HashSet<&str> = [
        "the", "a", "and", "of", "to", "in", "on", "for", "with", "at", "by", "an", "be", "is",
        "are", "was", "were", "it", "that", "this", "from", "as", "at", "but", "not", "or", "will",
        "has", "have", "its", "his", "her", "their", "they", "who", "which", "also", "been", "by",
        "an", "about",
    ]
    .iter()
    .cloned()
    .collect();

    let train_file_path = "example/dbpedia/dbpedia_csv/train.csv";
    let weights_file_path = "example/dbpedia/dbpedia_weights.bin";
    let vocab_file_path = "example/dbpedia/dbpedia_vocab.txt";

    let vocab_size = 2000;
    let num_experts = 14;
    let field_size = vocab_size + num_experts; // 2014 neurons

    let mut vocab_map: HashMap<String, usize> = HashMap::new();
    let mut vocab_list: Vec<String> = Vec::new();

    // 1. Check if we can load pre-trained weights and vocabulary
    let field = NeuronField::new(field_size);

    if Path::new(weights_file_path).exists() && Path::new(vocab_file_path).exists() {
        println!("Loading pre-trained model weights and vocabulary...");

        // Load vocabulary
        let vocab_content = std::fs::read_to_string(vocab_file_path)?;
        for (idx, line) in vocab_content.lines().enumerate() {
            vocab_map.insert(line.to_string(), idx);
            vocab_list.push(line.to_string());
        }

        // Load raw memory weights
        let bytes = std::fs::read(weights_file_path)?;
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                field.storage as *mut u8,
                field_size * std::mem::size_of::<neuron_poc::memory::GuardedNeuron>(),
            );
        }
        println!("Model loaded successfully in < 1ms!\n");
    } else {
        println!("Pre-trained model not found. Starting training on 560,000 samples...");
        println!("(This will take about 20 seconds and will save the weights for instant future startups)\n");

        if !Path::new(train_file_path).exists() {
            println!("Error: Training dataset not found!");
            println!("Please run 'mise run download_data' first to download the dataset.");
            return Ok(());
        }

        println!("--- Step 1: Building Vocabulary from 560,000 Training Samples ---");
        let mut rdr = csv::Reader::from_path(train_file_path)?;
        let mut word_counts: HashMap<String, [u32; 14]> = HashMap::new();

        for result in rdr.records() {
            let record = result?;
            let class_index: u32 = record[0].parse()?;
            let cat_idx = class_index - 1;
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

        let mut word_list: Vec<(String, [u32; 14], u32)> = word_counts
            .into_iter()
            .map(|(word, counts)| {
                let total_count: u32 = counts.iter().sum();
                (word, counts, total_count)
            })
            .collect();

        word_list.sort_by(|a, b| b.2.cmp(&a.2));
        let final_vocab: Vec<(String, [u32; 14])> = word_list
            .into_iter()
            .take(vocab_size)
            .map(|(word, counts, _)| (word, counts))
            .collect();

        // Save vocabulary to disk
        let mut vocab_file = File::create(vocab_file_path)?;
        for (word, _) in &final_vocab {
            writeln!(vocab_file, "{}", word)?;
        }

        // Load into memory maps
        for (idx, (word, _)) in final_vocab.iter().enumerate() {
            vocab_map.insert(word.clone(), idx);
            vocab_list.push(word.clone());
        }

        // Configure word neurons
        unsafe {
            for i in 0..vocab_size {
                let n = field.get_neuron(i);
                n.potential = 0.0;
                n.threshold = 1.0;

                let counts = final_vocab[i].1;
                let mut max_idx = 0;
                let mut max_val = 0;
                for (idx, &val) in counts.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = idx;
                    }
                }

                n.target_id = (vocab_size + max_idx) as u32;
                n.weight = 1.5;
            }

            for i in vocab_size..field_size {
                let n = field.get_neuron(i);
                n.potential = 0.0;
                n.threshold = 1.0;
                n.target_id = 999;
                n.weight = 0.0;
            }
        }

        println!("--- Step 2: Training on 560,000 Samples (Trainer Mode) ---");
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
                        let target_expert = word_neuron.target_id - vocab_size as u32;

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

        // Save raw memory weights to disk (Pointerless serialization!)
        println!("Saving model weights to disk for instant future startups...");
        let bytes = unsafe {
            std::slice::from_raw_parts(
                field.storage as *const u8,
                field_size * std::mem::size_of::<neuron_poc::memory::GuardedNeuron>(),
            )
        };
        std::fs::write(weights_file_path, bytes)?;
        println!("Model saved successfully!\n");
    }

    // 2. Interactive CLI Loop
    println!("--------------------------------------------------------------------");
    println!("Type any sentence or description below to classify it.");
    println!("The engine will route the context to the 14 experts in real-time.");
    println!("Type 'exit' or 'quit' to close the tool.");
    println!("--------------------------------------------------------------------\n");

    let mut input = String::new();
    loop {
        print!("👉 Enter text: ");
        io::stdout().flush()?;
        input.clear();
        io::stdin().read_line(&mut input)?;

        let trimmed = input.trim();
        if trimmed == "exit" || trimmed == "quit" {
            break;
        }

        if trimmed.is_empty() {
            continue;
        }

        let tokens = tokenize(trimmed);

        // Reset expert potentials
        unsafe {
            for i in vocab_size..field_size {
                field.get_neuron(i).potential = 0.0;
            }
        }

        // Present each word in Run Mode
        let mut recognized_words = Vec::new();
        for token in &tokens {
            if let Some(&word_idx) = vocab_map.get(token) {
                recognized_words.push(token.clone());
                unsafe {
                    let word_neuron = field.get_neuron(word_idx);
                    let target_expert = word_neuron.target_id;
                    let expert_neuron = field.get_neuron(target_expert as usize);
                    expert_neuron.potential += word_neuron.weight;
                }
            }
        }

        if recognized_words.is_empty() {
            println!("  ⚠️  None of the words were recognized in the 2,000-word vocabulary.");
            println!("      Try using more general descriptive words!\n");
            continue;
        }

        println!("  Recognized Vocab: {:?}", recognized_words);
        println!("  Expert Activations:");

        // Find the winning expert
        let mut predicted_idx = 0;
        let mut max_potential = -1.0;
        unsafe {
            for idx in 0..14 {
                let p = field.get_neuron(vocab_size + idx).potential;
                if p > max_potential {
                    max_potential = p;
                    predicted_idx = idx;
                }

                let cat = DBpediaCategory::from_class_index((idx + 1) as u32);
                let mut cat_name = cat.name().to_string();
                if cat_name.len() > 22 {
                    cat_name.truncate(22);
                }
                println!(
                    "    [{:22}]: {:.2} {}",
                    cat_name,
                    p,
                    "*".repeat((p * 10.0) as usize)
                );
            }
        }

        let winner = DBpediaCategory::from_class_index((predicted_idx + 1) as u32);
        println!(
            "\n  🏆 Winning Category: **{}** 🏆\n",
            winner.name().to_string().to_uppercase()
        );
        println!("--------------------------------------------------------------------");
    }

    println!("\nThank you for using the DBpedia Ontology CLI Tool! Goodbye!");
    Ok(())
}
