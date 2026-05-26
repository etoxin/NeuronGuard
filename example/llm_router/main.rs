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

use neuron_poc::neuron_guard::{ParallelRouter, ThreadBoundedNeuronField};
use rand::seq::SliceRandom;
use rand::Rng;
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;

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
        "Total Neuron Field Size: {} neurons (64 bytes each, aligned to 64 bytes)\n",
        field_size
    );

    // 2. Initialize the ThreadBoundedNeuronField
    let field = ThreadBoundedNeuronField::new(field_size);

    // Configure word neurons (0..100) to target their respective experts (100..105)
    unsafe {
        for i in 0..num_words {
            let n = field.get_neuron(i);
            n.token_id = i as u32;

            // Map word category to expert ID
            let category_idx = i / 20;
            let target_expert = (num_words + category_idx) as u32;

            // Add initial connection with high weight modifier
            n.update_or_add_connection(target_expert, 15);
        }

        // Configure Expert neurons (100..105)
        for i in num_words..field_size {
            let n = field.get_neuron(i);
            n.token_id = i as u32;
            n.active_connections = 0;
        }
    }

    // Set up the parallel router and accumulators
    let accumulators = Arc::new(
        (0..field_size)
            .map(|_| AtomicI32::new(0))
            .collect::<Vec<_>>(),
    );
    let router = ParallelRouter::new(Arc::clone(&accumulators));

    println!("--- Step 1: Testing Routing BEFORE Training (Run Mode) ---");
    let test_sentences = vec![
        ("rust compiler async bug", ExpertCategory::Technical),
        ("hello friend welcome buddy", ExpertCategory::Greeting),
        ("buy stock market profit", ExpertCategory::Financial),
        ("patient doctor medicine hospital", ExpertCategory::Medical),
        ("court judge lawyer contract", ExpertCategory::Legal),
    ];

    for (sentence, expected_cat) in &test_sentences {
        route_and_visualize(
            &field,
            &router,
            &accumulators,
            sentence,
            *expected_cat,
            num_words,
        );
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

            // Train on each word in the sentence using the Guard/Lease pattern
            for word in &chosen_words {
                if let Some(word_idx) = vocab.iter().position(|w| w == word) {
                    if let Some(lease) = field.try_acquire_lease(word_idx) {
                        let neuron = lease.neuron();
                        let correct_expert = (num_words + *category as usize) as u32;

                        // Amplify correct expert pathway
                        neuron.update_or_add_connection(correct_expert, 5);

                        // Suppress incorrect expert pathways
                        for i in 0..neuron.active_connections as usize {
                            let target = neuron.target_neuron_ids[i];
                            if target != correct_expert && target >= num_words as u32 {
                                neuron.weight_modifiers[i] =
                                    neuron.weight_modifiers[i].saturating_sub(15);
                            }
                        }
                    }
                }
            }
        }

        // Print progress every 5 epochs
        if epoch % 5 == 0 {
            unsafe {
                let mut avg_weight = 0.0;
                let mut total_connections = 0;
                for i in 0..num_words {
                    let n = field.get_neuron(i);
                    total_connections += n.active_connections;
                    for j in 0..n.active_connections as usize {
                        avg_weight += n.weight_modifiers[j] as f32;
                    }
                }
                avg_weight /= total_connections as f32;
                println!(
                    "  Epoch {:02}/20 Complete. Avg Connection Weight: {:.4}, Total Connections: {}",
                    epoch, avg_weight, total_connections
                );
            }
        }
    }

    println!("\n--- Step 3: Testing Routing AFTER Training (Run Mode) ---");
    for (sentence, expected_cat) in &test_sentences {
        route_and_visualize(
            &field,
            &router,
            &accumulators,
            sentence,
            *expected_cat,
            num_words,
        );
    }

    println!("\n--- Step 4: Routing a Complex Mixed Sentence ---");
    // Let's route a highly complex mixed sentence:
    // "patient medicine buy stock court"
    // This contains:
    // - Medical words: "patient", "medicine" (2 words)
    // - Financial words: "buy", "stock" (2 words)
    // - Legal words: "court" (1 word)
    // We expect Medical and Financial experts to compete and have the highest potentials!
    route_and_visualize(
        &field,
        &router,
        &accumulators,
        "patient medicine buy stock court",
        ExpertCategory::Medical,
        num_words,
    );

    println!("====================================================================");
}

fn route_and_visualize(
    field: &ThreadBoundedNeuronField,
    router: &ParallelRouter,
    accumulators: &Arc<Vec<AtomicI32>>,
    sentence: &str,
    expected_cat: ExpertCategory,
    num_words: usize,
) {
    let words: Vec<&str> = sentence.split_whitespace().collect();

    // Reset expert potentials in the accumulators
    for i in num_words..num_words + 5 {
        accumulators[i].store(0, Ordering::Relaxed);
    }

    // Present each word in Run Mode using the ParallelRouter
    for word in &words {
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
                let neuron = *field.get_neuron(word_idx);
                router.broadcast(neuron);
            }
        }
    }

    // Wait a tiny bit for the parallel threads to finish processing
    std::thread::sleep(std::time::Duration::from_millis(5));

    println!(
        "Sentence: \"{}\" (Expected: {})",
        sentence,
        expected_cat.name()
    );
    for i in 0..5 {
        let p = accumulators[num_words + i].load(Ordering::Relaxed);
        let cat_name = match i {
            0 => "Technical",
            1 => "Greeting ",
            2 => "Financial",
            3 => "Medical  ",
            4 => "Legal    ",
            _ => unreachable!(),
        };
        let bar_len = if p > 0 { p as usize } else { 0 };
        println!(
            "  [{}] Potential: {:3} {}",
            cat_name,
            p,
            "*".repeat(bar_len)
        );
    }
    println!();
}
