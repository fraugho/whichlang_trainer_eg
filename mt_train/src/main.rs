use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::error::Error;
use std::time::Instant;
use rand::seq::SliceRandom;
use rand::rng;
use csv::Reader;
use rand::prelude::IndexedMutRandom;

// Configuration for training
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub learning_rate: f32,
    pub epochs: usize,
    pub regularization: f32,
    pub dimension: usize,
    pub train_test_split: f32, // 0.8 means 80% training, 20% testing
    pub batch_size: usize,
    pub early_stopping_patience: usize,
    pub samples_per_language: usize, // New: equal samples per language
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            epochs: 100,
            regularization: 0.001,
            dimension: 4096,
            train_test_split: 0.8,
            batch_size: 32,
            early_stopping_patience: 10,
            samples_per_language: 1000, // Default to 1000 samples per language
        }
    }
}

// Training example
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TrainingExample {
    pub id: u32,
    pub lan_code: String,
    pub sentence: String,
}

// Feature extraction (copy from original code)
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Feature {
    AsciiNGram(u32),
    Unicode(char),
    UnicodeClass(char),
}

const SEED: u32 = 3_242_157_231u32;
const BIGRAM_MASK: u32 = (1 << 16) - 1;
const TRIGRAM_MASK: u32 = (1 << 24) - 1;

// Japanese/CJK Unicode ranges
const JP_PUNCT_START: u32 = 0x3000;
const JP_PUNCT_END: u32 = 0x303f;
const JP_HIRAGANA_START: u32 = 0x3040;
const JP_HIRAGANA_END: u32 = 0x309f;
const JP_KATAKANA_START: u32 = 0x30a0;
const JP_KATAKANA_END: u32 = 0x30ff;
const CJK_KANJI_START: u32 = 0x4e00;
const CJK_KANJI_END: u32 = 0x9faf;
const JP_HALFWIDTH_KATAKANA_START: u32 = 0xff61;
const JP_HALFWIDTH_KATAKANA_END: u32 = 0xff90;

#[inline(always)]
fn murmurhash2(mut k: u32, seed: u32) -> u32 {
    const M: u32 = 0x5bd1_e995;
    let mut h: u32 = seed;
    k = k.wrapping_mul(M);
    k ^= k >> 24;
    k = k.wrapping_mul(M);
    h = h.wrapping_mul(M);
    h ^= k;
    h ^= h >> 13;
    h = h.wrapping_mul(M);
    h ^ (h >> 15)
}

impl Feature {
    #[inline(always)]
    pub fn to_hash(&self) -> u32 {
        match self {
            Feature::AsciiNGram(ngram) => murmurhash2(*ngram, SEED),
            Feature::Unicode(chr) => murmurhash2(*chr as u32 / 128, SEED ^ 2),
            Feature::UnicodeClass(chr) => murmurhash2(classify_codepoint(*chr), SEED ^ 4),
        }
    }
}

fn classify_codepoint(chr: char) -> u32 {
    [
        160, 161, 171, 172, 173, 174, 187, 192, 196, 199, 200, 201, 202, 205, 214, 220, 223,
        224, 225, 226, 227, 228, 231, 232, 233, 234, 235, 236, 237, 238, 239, 242, 243, 244,
        245, 246, 249, 250, 251, 252, 333, 339,
        JP_PUNCT_START, JP_PUNCT_END, JP_HIRAGANA_START, JP_HIRAGANA_END,
        JP_KATAKANA_START, JP_KATAKANA_END, CJK_KANJI_START, CJK_KANJI_END,
        JP_HALFWIDTH_KATAKANA_START, JP_HALFWIDTH_KATAKANA_END,
    ]
    .binary_search(&(chr as u32))
    .unwrap_or_else(|pos| pos) as u32
}

pub fn emit_tokens(text: &str, mut listener: impl FnMut(Feature)) {
    let mut prev = ' ' as u32;
    let mut num_previous_ascii_chr = 1;
    for chr in text.chars() {
        let code = chr.to_ascii_lowercase() as u32;
        if !chr.is_ascii() {
            listener(Feature::Unicode(chr));
            listener(Feature::UnicodeClass(chr));
            num_previous_ascii_chr = 0;
            continue;
        }
        prev = prev << 8 | code;
        match num_previous_ascii_chr {
            0 => {
                num_previous_ascii_chr = 1;
            }
            1 => {
                listener(Feature::AsciiNGram(prev & BIGRAM_MASK));
                num_previous_ascii_chr = 2;
            }
            2 => {
                listener(Feature::AsciiNGram(prev & BIGRAM_MASK));
                listener(Feature::AsciiNGram(prev & TRIGRAM_MASK));
                num_previous_ascii_chr = 3;
            }
            3 => {
                listener(Feature::AsciiNGram(prev & BIGRAM_MASK));
                listener(Feature::AsciiNGram(prev & TRIGRAM_MASK));
                listener(Feature::AsciiNGram(prev));
            }
            _ => {
                unreachable!();
            }
        }
        if !chr.is_alphanumeric() {
            prev = ' ' as u32;
        }
    }
}

fn format_duration(seconds: f64) -> String {
    let hours = (seconds / 3600.0) as u32;
    let minutes = ((seconds % 3600.0) / 60.0) as u32;
    let secs = (seconds % 60.0) as u32;
    
    if hours > 0 {
        format!("{}h {:02}m {:02}s", hours, minutes, secs)
    } else if minutes > 0 {
        format!("{}m {:02}s", minutes, secs)
    } else {
        format!("{}s", secs)
    }
}

// Main trainer struct
pub struct LanguageDetectorTrainer {
    pub language_codes: Vec<String>,
    pub language_names: HashMap<String, String>,
    pub weights: Vec<f32>,
    pub intercepts: Vec<f32>,
    pub config: TrainingConfig,
}

impl LanguageDetectorTrainer {
    pub fn new(language_codes: Vec<String>, language_names: HashMap<String, String>, config: TrainingConfig) -> Self {
        let num_languages = language_codes.len();
        let total_weights = config.dimension * num_languages;
        
        // Initialize weights with small random values
        let mut rng = rng();
        let weights: Vec<f32> = (0..total_weights)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.01)
            .collect();
        
        let intercepts: Vec<f32> = (0..num_languages)
            .map(|_| (rand::random::<f32>() - 0.5) * 0.01)
            .collect();

        Self {
            language_codes,
            language_names,
            weights,
            intercepts,
            config,
        }
    }

    // Load training data from CSV
    pub fn load_csv_data(file_path: &str) -> Result<Vec<TrainingExample>, Box<dyn Error>> {
        let mut file = Reader::from_path(file_path)?;
        let mut examples = Vec::new();
        for result in file.deserialize() {
            let record: TrainingExample = result?;
            examples.push(record);
        }
        println!("Loaded {} training examples", examples.len());
        Ok(examples)
    }

    // NEW: Create balanced dataset with equal samples per language
    pub fn create_balanced_dataset(&self, data: &[TrainingExample]) -> Vec<TrainingExample> {
        let mut rng = rng();
        let mut lang_data: HashMap<String, Vec<TrainingExample>> = HashMap::new();
        
        // Group data by language
        for example in data {
            lang_data.entry(example.lan_code.clone())
                .or_insert_with(Vec::new)
                .push(example.clone());
        }

        let mut balanced_data = Vec::new();
        let mut total_samples = 0;
        let mut upsampled_languages = Vec::new();
        let mut downsampled_languages = Vec::new();

        println!("\nCreating balanced dataset with {} samples per language:", self.config.samples_per_language);
        
        for lang_code in &self.language_codes {
            if let Some(examples) = lang_data.get_mut(lang_code) {
                let original_count = examples.len();
                
                if original_count >= self.config.samples_per_language {
                    // Downsample: randomly select samples
                    examples.shuffle(&mut rng);
                    examples.truncate(self.config.samples_per_language);
                    downsampled_languages.push((lang_code.clone(), original_count));
                } else {
                    // Upsample: duplicate examples with slight variations
                    let mut upsampled = Vec::new();
                    let needed = self.config.samples_per_language;
                    
                    // First add all original examples
                    upsampled.extend_from_slice(examples);
                    
                    // Then duplicate randomly to reach target
                    while upsampled.len() < needed {
                        let random_example = examples.choose_mut(&mut rng).unwrap().clone();
                        upsampled.push(random_example);
                    }
                    
                    // Ensure exact count
                    upsampled.truncate(needed);
                    *examples = upsampled;
                    upsampled_languages.push((lang_code.clone(), original_count));
                }
                
                balanced_data.extend_from_slice(examples);
                total_samples += examples.len();
                
                let lang_name = self.language_names.get(lang_code).unwrap_or(lang_code);
                println!("  {}: {} -> {} samples ({})", 
                        lang_code, original_count, examples.len(), lang_name);
            }
        }

        println!("\nBalancing summary:");
        println!("  Total languages: {}", self.language_codes.len());
        println!("  Total samples: {} ({}k per language)", total_samples, self.config.samples_per_language);
        println!("  Upsampled languages: {}", upsampled_languages.len());
        println!("  Downsampled languages: {}", downsampled_languages.len());
        
        if !upsampled_languages.is_empty() {
            println!("\nUpsampled languages (original -> target):");
            for (lang, original) in &upsampled_languages {
                let lang_name = self.language_names.get(lang).unwrap_or(lang);
                println!("  {}: {} -> {} ({})", lang, original, self.config.samples_per_language, lang_name);
            }
        }
        
        if !downsampled_languages.is_empty() {
            println!("\nDownsampled languages (original -> target):");
            for (lang, original) in &downsampled_languages {
                let lang_name = self.language_names.get(lang).unwrap_or(lang);
                println!("  {}: {} -> {} ({})", lang, original, self.config.samples_per_language, lang_name);
            }
        }

        // Final shuffle
        balanced_data.shuffle(&mut rng);
        balanced_data
    }

    // Helper function to convert language code to C++ enum name
    fn lang_code_to_cpp_enum(code: &str) -> String {
        // Convert language code to enum variant (capitalize first letter)
        code.chars()
            .enumerate()
            .map(|(i, c)| if i == 0 { c.to_uppercase().collect::<String>() } else { c.to_string() })
            .collect::<String>()
    }
    pub fn extract_features(&self, text: &str) -> HashMap<u32, f32> {
        let mut feature_counts = HashMap::new();
        let mut total_features = 0u32;
        
        emit_tokens(text, |feature| {
            total_features += 1;
            let hash = feature.to_hash();
            let bucket = hash % self.config.dimension as u32;
            *feature_counts.entry(bucket).or_insert(0.0) += 1.0;
        });

        // Normalize by sqrt of total features (matching original code)
        if total_features > 0 {
            let norm_factor = 1.0 / (total_features as f32).sqrt();
            for count in feature_counts.values_mut() {
                *count *= norm_factor;
            }
        }

        feature_counts
    }

    // Predict language scores
    pub fn predict(&self, features: &HashMap<u32, f32>) -> Vec<f32> {
        let mut scores = self.intercepts.clone();
        
        for (&bucket, &count) in features {
            let weight_start = bucket as usize * self.language_codes.len();
            for (lang_idx, score) in scores.iter_mut().enumerate() {
                if weight_start + lang_idx < self.weights.len() {
                    *score += self.weights[weight_start + lang_idx] * count;
                }
            }
        }
        
        scores
    }

    // Softmax function
    pub fn softmax(scores: &[f32]) -> Vec<f32> {
        let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum_exp: f32 = exp_scores.iter().sum();
        if sum_exp > 0.0 {
            exp_scores.iter().map(|&s| s / sum_exp).collect()
        } else {
            vec![1.0 / scores.len() as f32; scores.len()]
        }
    }

    // Training step
    pub fn train_step(&mut self, examples: &[TrainingExample]) -> f32 {
        let mut total_loss = 0.0;
        let mut processed = 0;

        for example in examples {
            if let Some(target_idx) = self.language_codes.iter().position(|code| code == &example.lan_code) {
                let features = self.extract_features(&example.sentence);
                if features.is_empty() {
                    continue;
                }

                let scores = self.predict(&features);
                let probabilities = Self::softmax(&scores);
                
                // Cross-entropy loss
                let loss = -probabilities[target_idx].max(1e-10).ln();
                total_loss += loss;
                processed += 1;

                // Gradient descent
                for (&bucket, &feature_count) in &features {
                    let weight_start = bucket as usize * self.language_codes.len();
                    
                    for (lang_idx, &prob) in probabilities.iter().enumerate() {
                        if weight_start + lang_idx < self.weights.len() {
                            let gradient = if lang_idx == target_idx {
                                feature_count * (prob - 1.0)
                            } else {
                                feature_count * prob
                            };
                            
                            // Update with L2 regularization
                            let weight_idx = weight_start + lang_idx;
                            let reg_term = self.config.regularization * self.weights[weight_idx];
                            self.weights[weight_idx] -= self.config.learning_rate * (gradient + reg_term);
                        }
                    }
                }

                // Update intercepts
                for (lang_idx, &prob) in probabilities.iter().enumerate() {
                    let gradient = if lang_idx == target_idx {
                        prob - 1.0
                    } else {
                        prob
                    };
                    
                    self.intercepts[lang_idx] -= self.config.learning_rate * gradient;
                }
            }
        }

        if processed > 0 {
            total_loss / processed as f32
        } else {
            0.0
        }
    }

    // Full training loop
    pub fn train(&mut self, training_data: &[TrainingExample]) {
        let mut rng = rng();
        let mut best_loss = f32::INFINITY;
        let mut patience_counter = 0;

        // Create balanced dataset first
        let balanced_data = self.create_balanced_dataset(training_data);
        
        // Split balanced data
        let mut shuffled_data = balanced_data;
        shuffled_data.shuffle(&mut rng);
        
        let split_idx = (shuffled_data.len() as f32 * self.config.train_test_split) as usize;
        let (train_data, test_data) = shuffled_data.split_at(split_idx);
        
        println!("\nTraining on {} examples, testing on {} examples", train_data.len(), test_data.len());

        let start_time = Instant::now();

        for epoch in 0..self.config.epochs {
            let epoch_start = Instant::now();
            let mut epoch_data = train_data.to_vec();
            epoch_data.shuffle(&mut rng);

            // Process in batches
            let mut total_loss = 0.0;
            let mut num_batches = 0;
            
            for batch in epoch_data.chunks(self.config.batch_size) {
                let loss = self.train_step(batch);
                total_loss += loss;
                num_batches += 1;
            }

            let avg_loss = if num_batches > 0 { total_loss / num_batches as f32 } else { 0.0 };
            
            // Calculate ETA
            let elapsed = start_time.elapsed().as_secs_f64();
            let avg_time_per_epoch = elapsed / (epoch + 1) as f64;
            let remaining_epochs = self.config.epochs - (epoch + 1);
            let eta_seconds = avg_time_per_epoch * remaining_epochs as f64;
            
            // Evaluate on test data every 10 epochs
            if epoch % 10 == 0 || epoch == self.config.epochs - 1 {
                let test_accuracy = self.evaluate(test_data);
                println!("Epoch {}: Avg Loss = {:.4}, Test Accuracy = {:.2}% | ETA: {}", 
                        epoch + 1, avg_loss, test_accuracy * 100.0, format_duration(eta_seconds));
            } else {
                println!("Epoch {}: Avg Loss = {:.4} | ETA: {}", 
                        epoch + 1, avg_loss, format_duration(eta_seconds));
            }

            // Early stopping
            if avg_loss < best_loss {
                best_loss = avg_loss;
                patience_counter = 0;
            } else {
                patience_counter += 1;
                if patience_counter >= self.config.early_stopping_patience {
                    println!("Early stopping at epoch {}", epoch + 1);
                    break;
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        println!("Training completed in {}", format_duration(total_time));
    }

    // Evaluate model
    pub fn evaluate(&self, test_data: &[TrainingExample]) -> f32 {
        let mut correct = 0;
        let mut total = 0;

        for example in test_data {
            if let Some(target_idx) = self.language_codes.iter().position(|code| code == &example.lan_code) {
                let features = self.extract_features(&example.sentence);
                if !features.is_empty() {
                    let scores = self.predict(&features);
                    let predicted_idx = scores.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    
                    if predicted_idx == target_idx {
                        correct += 1;
                    }
                    total += 1;
                }
            }
        }

        if total > 0 {
            correct as f32 / total as f32
        } else {
            0.0
        }
    }

    // Export weights to C++ header file
    pub fn export_weights(&self, output_file: &str) -> Result<(), Box<dyn Error>> {
        let mut file = File::create(output_file)?;
        
        writeln!(file, "// Auto-generated language detection weights")?;
        writeln!(file, "// Generated from {} languages with {} features", 
                self.language_codes.len(), self.config.dimension)?;
        writeln!(file, "// Trained with {} samples per language (egalitarian)", 
                self.config.samples_per_language)?;
        writeln!(file, "#pragma once")?;
        writeln!(file, "#include <array>")?;
        writeln!(file, "#include <string>")?;
        writeln!(file)?;

        // Generate enum for languages
        writeln!(file, "enum class Lang {{")?;
        for code in &self.language_codes {
            let enum_name = Self::lang_code_to_cpp_enum(code);
            writeln!(file, "    {},  // {}", enum_name, 
                    self.language_names.get(code).unwrap_or(code))?;
        }
        writeln!(file, "}};")?;
        writeln!(file)?;

        // Generate three_letter_code function
        writeln!(file, "std::string three_letter_code(Lang language) {{")?;
        writeln!(file, "    switch (language) {{")?;
        for code in &self.language_codes {
            let enum_name = Self::lang_code_to_cpp_enum(code);
            writeln!(file, "        case Lang::{}: return \"{}\";", enum_name, code)?;
        }
        writeln!(file, "    }}")?;
        writeln!(file, "    return \"unknown\";")?;
        writeln!(file, "}}")?;
        writeln!(file)?;

        // Generate languages array
        writeln!(file, "const std::array<Lang, {}> LANGUAGES = {{", self.language_codes.len())?;
        for code in &self.language_codes {
            let enum_name = Self::lang_code_to_cpp_enum(code);
            writeln!(file, "    Lang::{},", enum_name)?;
        }
        writeln!(file, "}};")?;
        writeln!(file)?;

        // Generate weights array
        writeln!(file, "const std::array<float, {}> WEIGHTS = {{", self.weights.len())?;
        for (i, &weight) in self.weights.iter().enumerate() {
            if i % 8 == 0 {
                write!(file, "    ")?;
            }
            write!(file, "{:.6}f", weight)?;
            if i < self.weights.len() - 1 {
                write!(file, ",")?;
            }
            if i % 8 == 7 || i == self.weights.len() - 1 {
                writeln!(file)?;
            } else {
                write!(file, " ")?;
            }
        }
        writeln!(file, "}};")?;
        writeln!(file)?;

        // Generate intercepts array
        writeln!(file, "const float INTERCEPTS[{}] = {{", self.intercepts.len())?;
        for (i, &intercept) in self.intercepts.iter().enumerate() {
            if i % 8 == 0 {
                write!(file, "    ")?;
            }
            write!(file, "{:.6}f", intercept)?;
            if i < self.intercepts.len() - 1 {
                write!(file, ",")?;
            }
            if i % 8 == 7 || i == self.intercepts.len() - 1 {
                writeln!(file)?;
            } else {
                write!(file, " ")?;
            }
        }
        writeln!(file, "}};")?;

        println!("Weights exported to {}", output_file);
        Ok(())
    }

    // Print training statistics
    pub fn print_language_stats(&self, data: &[TrainingExample]) {
        let mut lang_counts: HashMap<String, usize> = HashMap::new();
        for example in data {
            *lang_counts.entry(example.lan_code.clone()).or_insert(0) += 1;
        }

        println!("\nOriginal language distribution in dataset:");
        let mut sorted_langs: Vec<_> = lang_counts.iter().collect();
        sorted_langs.sort_by(|a, b| b.1.cmp(a.1));
        
        for (code, count) in sorted_langs.iter().take(20) { // Show top 20
            let name = self.language_names.get(*code).unwrap_or(code);
            println!("  {}: {} ({} examples)", code, name, count);
        }
        
        if sorted_langs.len() > 20 {
            println!("  ... and {} more languages", sorted_langs.len() - 20);
        }
    }
}

// Main function to run training
fn main() -> Result<(), Box<dyn Error>> {
    // Load language mappings
    let language_names = load_language_mappings("../dataset/lan_to_language.json")?;
    
    // Load training data
    let training_data = LanguageDetectorTrainer::load_csv_data("../dataset/sentences.csv")?;
    
    // Get unique language codes from data
    let mut language_codes: Vec<String> = training_data.iter()
        .map(|ex| ex.lan_code.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    language_codes.sort();
    
    println!("Found {} unique languages", language_codes.len());

    // Configure training with egalitarian sampling
    let config = TrainingConfig {
        learning_rate: 0.01,
        epochs: 200,
        regularization: 0.001,
        dimension: 4096,
        train_test_split: 0.8,
        batch_size: 64,
        early_stopping_patience: 20,
        samples_per_language: 1000, // Equal samples for all languages
    };

    // Create and train model
    let mut trainer = LanguageDetectorTrainer::new(language_codes, language_names, config);
    trainer.print_language_stats(&training_data);
    
    println!("\nStarting egalitarian training...");
    trainer.train(&training_data);
    
    // Export results
    trainer.export_weights("weights_balanced.rs")?;
    
    println!("Egalitarian training completed successfully!");
    Ok(())
}

// Helper function to load language mappings
fn load_language_mappings(file_path: &str) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let file_content = std::fs::read_to_string(file_path)?;
    let mappings: HashMap<String, String> = serde_json::from_str(&file_content)?;
    Ok(mappings)
}
