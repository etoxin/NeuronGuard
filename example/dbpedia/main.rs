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

use neuronguard::neuron_guard::{ThreadBoundedNeuron, ThreadBoundedNeuronField};
use neuronguard::run::{evaluate_neuron_potentials, tokenize};
use neuronguard::train::train_neuron_connection;
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
    let field = ThreadBoundedNeuronField::new(field_size);

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
                field_size * std::mem::size_of::<ThreadBoundedNeuron>(),
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
                n.token_id = i as u32;

                let counts = final_vocab[i].1;
                let mut max_idx = 0;
                let mut max_val = 0;
                for (idx, &val) in counts.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = idx;
                    }
                }

                let target_expert = (vocab_size + max_idx) as u32;
                n.update_or_add_connection(target_expert, 15);
            }

            for i in vocab_size..field_size {
                let n = field.get_neuron(i);
                n.token_id = i as u32;
                n.active_connections = 0;
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
                    let correct_expert = (vocab_size + category as usize) as u32;
                    train_neuron_connection(&field, word_idx, correct_expert, vocab_size, 5, 15);
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
                field_size * std::mem::size_of::<ThreadBoundedNeuron>(),
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
        let mut expert_potentials = [0i32; 14];

        // Present each word in Run Mode
        let mut recognized_words = Vec::new();
        for token in &tokens {
            if let Some(&word_idx) = vocab_map.get(token) {
                recognized_words.push(token.clone());
                evaluate_neuron_potentials(
                    &field,
                    word_idx,
                    vocab_size,
                    field_size,
                    &mut expert_potentials,
                );
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
        let mut max_potential = i32::MIN;
        for idx in 0..14 {
            let p = expert_potentials[idx];
            if p > max_potential {
                max_potential = p;
                predicted_idx = idx;
            }

            let cat = DBpediaCategory::from_class_index((idx + 1) as u32);
            let mut cat_name = cat.name().to_string();
            if cat_name.len() > 22 {
                cat_name.truncate(22);
            }
            let bar_len = if p > 0 { p as usize } else { 0 };
            println!("    [{:22}]: {:3} {}", cat_name, p, "*".repeat(bar_len));
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
