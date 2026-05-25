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
use rand::seq::SliceRandom;
use rand::Rng;

// Define our 5 Expert categories
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExpertCategory {
    Technical = 0,
    Greeting = 1,
    Financial = 2,
    Medical = 3,
    Legal = 4,
}

impl ExpertCategory {
    pub fn name(&self) -> &'static str {
        match self {
            ExpertCategory::Technical => "Technical",
            ExpertCategory::Greeting => "Greeting",
            ExpertCategory::Financial => "Financial",
            ExpertCategory::Medical => "Medical",
            ExpertCategory::Legal => "Legal",
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

fn main() {
    println!("====================================================================");
    println!("🧠 Large-Scale LLM Mixture-of-Experts (MoE) Router Simulation 🧠");
    println!("====================================================================\n");

    // 1. Define a 100-word vocabulary (20 words per category)
    let tech_words = vec![
        "rust", "code", "compiler", "bug", "memory", "pointer", "thread", "async", "cargo",
        "struct", "enum", "trait", "panic", "unsafe", "borrow", "lifetime", "macro", "crate",
        "vector", "string",
    ];
    let greet_words = vec![
        "hello",
        "greet",
        "welcome",
        "friend",
        "hi",
        "hey",
        "morning",
        "evening",
        "pleasure",
        "meet",
        "smile",
        "happy",
        "cheers",
        "thanks",
        "thank",
        "appreciate",
        "kind",
        "dear",
        "buddy",
        "awesome",
    ];
    let finance_words = vec![
        "buy",
        "stock",
        "market",
        "price",
        "finance",
        "business",
        "money",
        "trade",
        "investment",
        "bank",
        "cash",
        "profit",
        "loss",
        "revenue",
        "share",
        "dividend",
        "equity",
        "asset",
        "debt",
        "loan",
    ];
    let medical_words = vec![
        "patient", "doctor", "medicine", "health", "hospital", "nurse", "clinic", "disease",
        "cure", "drug", "pain", "surgery", "therapy", "virus", "vaccine", "blood", "heart",
        "brain", "fever", "cough",
    ];
    let legal_words = vec![
        "court",
        "judge",
        "lawyer",
        "contract",
        "legal",
        "law",
        "case",
        "trial",
        "jury",
        "appeal",
        "sue",
        "claim",
        "defense",
        "witness",
        "patent",
        "trademark",
        "copyright",
        "clause",
        "fine",
        "guilty",
    ];

    let mut vocab = Vec::new();
    vocab.extend(tech_words.clone());
    vocab.extend(greet_words.clone());
    vocab.extend(finance_words.clone());
    vocab.extend(medical_words.clone());
    vocab.extend(legal_words.clone());

    let num_words = vocab.len(); // 100 words
    let num_experts = 5;
    let field_size = num_words + num_experts; // 105 neurons

    println!("Vocabulary Size: {} words", num_words);
    println!("Number of Experts: {} experts", num_experts);
    println!(
        "Total Neuron Field Size: {} neurons (16 bytes each)\n",
        field_size
    );

    // 2. Initialize the NeuronField
    let field = NeuronField::new(field_size);

    // Configure word neurons (0..100) to target their respective experts (100..105)
    unsafe {
        for i in 0..num_words {
            let n = field.get_neuron(i);
            n.potential = 0.0;
            n.threshold = 1.0;

            // Map word category to expert ID
            let category_idx = i / 20;
            n.target_id = (num_words + category_idx) as u32;
            n.weight = 1.5; // Start with high weight (susceptible to noise)
        }

        // Configure Expert neurons (100..105)
        for i in num_words..field_size {
            let n = field.get_neuron(i);
            n.potential = 0.0;
            n.threshold = 1.0;
            n.target_id = 999; // End of chain
            n.weight = 0.0;
        }
    }

    println!("--- Step 1: Testing Routing BEFORE Training (Run Mode) ---");
    // Let's route a sentence from each category
    let test_sentences = vec![
        ("rust compiler async bug", ExpertCategory::Technical),
        ("hello friend welcome buddy", ExpertCategory::Greeting),
        ("buy stock market profit", ExpertCategory::Financial),
        ("patient doctor medicine hospital", ExpertCategory::Medical),
        ("court judge lawyer contract", ExpertCategory::Legal),
    ];

    for (sentence, expected_cat) in &test_sentences {
        route_and_visualize(&field, sentence, *expected_cat);
    }

    println!("\n--- Step 2: Training the Decent-Sized Model (Trainer Mode) ---");
    println!("Generating random sentences and applying the Guard feedback loop...");

    let mut rng = rand::thread_rng();
    let categories = vec![
        ExpertCategory::Technical,
        ExpertCategory::Greeting,
        ExpertCategory::Financial,
        ExpertCategory::Medical,
        ExpertCategory::Legal,
    ];

    for epoch in 1..=20 {
        // Generate 50 random training sentences per epoch (10 per category)
        for _ in 0..50 {
            let category = categories.choose(&mut rng).unwrap();
            let word_pool = match category {
                ExpertCategory::Technical => &tech_words,
                ExpertCategory::Greeting => &greet_words,
                ExpertCategory::Financial => &finance_words,
                ExpertCategory::Medical => &medical_words,
                ExpertCategory::Legal => &legal_words,
            };

            // Choose 3 to 5 random words from the pool
            let num_sentence_words = rng.gen_range(3..=5);
            let mut chosen_words = Vec::new();
            for _ in 0..num_sentence_words {
                chosen_words.push(*word_pool.choose(&mut rng).unwrap());
            }

            // Train on each word in the sentence
            for word in &chosen_words {
                if let Some(word_idx) = vocab.iter().position(|w| w == word) {
                    unsafe {
                        let word_neuron = field.get_neuron(word_idx);
                        let target_expert = word_neuron.target_id - num_words as u32;

                        // Apply feedback: positive if it targets the correct expert, negative if incorrect
                        let feedback = if target_expert == *category as u32 {
                            0.05
                        } else {
                            -0.15
                        };

                        propagate_trainer(&field, word_idx as u32, 1.0, None, feedback);
                    }
                }
            }
        }

        // Print progress every 5 epochs
        if epoch % 5 == 0 {
            unsafe {
                let mut avg_weight = 0.0;
                for i in 0..num_words {
                    avg_weight += field.get_neuron(i).weight;
                }
                avg_weight /= num_words as f32;
                println!(
                    "  Epoch {:02}/20 Complete. Average Word Weight: {:.4}",
                    epoch, avg_weight
                );
            }
        }
    }

    println!("\n--- Step 3: Testing Routing AFTER Training (Run Mode) ---");
    // Let's route the same sentences again to see the sharp routing!
    for (sentence, expected_cat) in &test_sentences {
        route_and_visualize(&field, sentence, *expected_cat);
    }

    println!("\n--- Step 4: Routing a Complex Mixed Sentence ---");
    // Let's route a highly complex mixed sentence:
    // "the patient needs medicine to buy stock in court"
    // This contains:
    // - Medical words: "patient", "medicine" (2 words)
    // - Financial words: "buy", "stock" (2 words)
    // - Legal words: "court" (1 word)
    // We expect Medical and Financial experts to compete and have the highest potentials!
    route_and_visualize(
        &field,
        "patient medicine buy stock court",
        ExpertCategory::Medical,
    );

    println!("====================================================================");
}

fn route_and_visualize(field: &NeuronField, sentence: &str, expected_cat: ExpertCategory) {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let num_words = 100;

    // Reset expert potentials
    unsafe {
        for i in num_words..105 {
            field.get_neuron(i).potential = 0.0;
        }
    }

    // Present each word in Run Mode
    for word in &words {
        // Find word index in our 100-word vocabulary
        let tech_words = vec![
            "rust", "code", "compiler", "bug", "memory", "pointer", "thread", "async", "cargo",
            "struct", "enum", "trait", "panic", "unsafe", "borrow", "lifetime", "macro", "crate",
            "vector", "string",
        ];
        let greet_words = vec![
            "hello",
            "greet",
            "welcome",
            "friend",
            "hi",
            "hey",
            "morning",
            "evening",
            "pleasure",
            "meet",
            "smile",
            "happy",
            "cheers",
            "thanks",
            "thank",
            "appreciate",
            "kind",
            "dear",
            "buddy",
            "awesome",
        ];
        let finance_words = vec![
            "buy",
            "stock",
            "market",
            "price",
            "finance",
            "business",
            "money",
            "trade",
            "investment",
            "bank",
            "cash",
            "profit",
            "loss",
            "revenue",
            "share",
            "dividend",
            "equity",
            "asset",
            "debt",
            "loan",
        ];
        let medical_words = vec![
            "patient", "doctor", "medicine", "health", "hospital", "nurse", "clinic", "disease",
            "cure", "drug", "pain", "surgery", "therapy", "virus", "vaccine", "blood", "heart",
            "brain", "fever", "cough",
        ];
        let legal_words = vec![
            "court",
            "judge",
            "lawyer",
            "contract",
            "legal",
            "law",
            "case",
            "trial",
            "jury",
            "appeal",
            "sue",
            "claim",
            "defense",
            "witness",
            "patent",
            "trademark",
            "copyright",
            "clause",
            "fine",
            "guilty",
        ];

        let mut vocab = Vec::new();
        vocab.extend(tech_words);
        vocab.extend(greet_words);
        vocab.extend(finance_words);
        vocab.extend(medical_words);
        vocab.extend(legal_words);

        if let Some(word_idx) = vocab.iter().position(|w| w == word) {
            unsafe {
                let word_neuron = field.get_neuron(word_idx);
                let target_expert = word_neuron.target_id;
                let expert_neuron = field.get_neuron(target_expert as usize);
                expert_neuron.potential += word_neuron.weight;
            }
        }
    }

    println!(
        "Sentence: \"{}\" (Expected: {})",
        sentence,
        expected_cat.name()
    );
    unsafe {
        for i in 0..5 {
            let p = field.get_neuron(num_words + i).potential;
            let cat_name = match i {
                0 => "Technical",
                1 => "Greeting ",
                2 => "Financial",
                3 => "Medical  ",
                4 => "Legal    ",
                _ => unreachable!(),
            };
            println!(
                "  [{}] Potential: {:.2} {}",
                cat_name,
                p,
                "*".repeat((p * 10.0) as usize)
            );
        }
    }
    println!();
}
