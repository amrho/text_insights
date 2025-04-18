use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

#[derive(Debug, Serialize, Deserialize)]
pub struct AnalysisResults {
    pub files_processed: usize,
    pub total_words: usize,
    pub unique_words: usize,
    pub avg_words_per_file: f64,
    pub top_words: Vec<(String, usize)>,
    pub word_frequencies: HashMap<String, usize>,
    pub character_frequencies: HashMap<char, usize>,
    pub sentence_lengths: Vec<usize>,
    pub readability_scores: HashMap<String, f64>,
    pub sentiment_scores: HashMap<String, f64>,
    pub sentiment_score: Option<f64>,
    pub sentence_count: usize,
    pub ngram_frequencies: HashMap<usize, Vec<(String, usize)>>,
    pub common_phrases: Vec<(String, usize)>,
    pub file_stats: Vec<FileStats>,
    pub pos_counts: HashMap<String, usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileStats {
    pub filename: String,
    pub word_count: usize,
    pub unique_words: usize,
    pub avg_word_length: f64,
    pub longest_word: String,
    pub sentiment_score: f64,
}

pub fn analyze_paths(paths: &[PathBuf], recursive: bool, pattern: Option<&str>) -> Result<AnalysisResults> {
    let files = collect_files(paths, recursive, pattern)?;
    
    if files.is_empty() {
        return Err(anyhow::anyhow!("No matching files found to analyze"));
    }
    
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} files ({eta})")
        .unwrap()
        .progress_chars("#>-"));
    
    // Process files in parallel
    let file_results: Vec<_> = files.par_iter()
        .map(|file| {
            let res = analyze_file(file);
            pb.inc(1);
            res
        })
        .collect::<Result<Vec<_>>>()?;
    
    pb.finish_with_message("Analysis complete!");
    
    // Aggregate results
    let mut total_words = 0;
    let mut word_frequencies = HashMap::new();
    let mut character_frequencies = HashMap::new();
    let mut sentence_lengths = Vec::new();
    let mut file_stats = Vec::new();
    let mut all_text = String::new();
    
    // N-gram storage (2-grams, 3-grams, 4-grams)
    let mut ngram_frequencies: HashMap<usize, HashMap<String, usize>> = HashMap::new();
    for n in 2..=4 {
        ngram_frequencies.insert(n, HashMap::new());
    }
    
    for result in &file_results {
        total_words += result.word_count;
        all_text.push_str(&result.full_text);
        all_text.push(' ');
        
        // Aggregate word frequencies
        for (word, count) in &result.word_frequencies {
            *word_frequencies.entry(word.clone()).or_insert(0) += count;
        }
        
        // Aggregate character frequencies
        for (ch, count) in &result.character_frequencies {
            *character_frequencies.entry(*ch).or_insert(0) += count;
        }
        
        // Collect sentence lengths
        sentence_lengths.extend(result.sentence_lengths.clone());
        
        // Aggregate n-grams
        for (n, ngram_map) in &result.ngram_frequencies {
            let global_map = ngram_frequencies.get_mut(n).unwrap();
            for (ngram, count) in ngram_map {
                *global_map.entry(ngram.clone()).or_insert(0) += count;
            }
        }
        
        // Add file stats
        file_stats.push(FileStats {
            filename: result.filename.clone(),
            word_count: result.word_count,
            unique_words: result.word_frequencies.len(),
            avg_word_length: result.avg_word_length,
            longest_word: result.longest_word.clone(),
            sentiment_score: result.sentiment_score,
        });
    }
    
    // Sort word frequencies to get top words
    let mut top_words: Vec<(String, usize)> = word_frequencies.iter()
        .map(|(word, count)| (word.clone(), *count))
        .collect();
    
    top_words.sort_by(|a, b| b.1.cmp(&a.1));
    
    // Prepare sorted n-grams for output
    let mut ngram_output = HashMap::new();
    for (n, freq_map) in &ngram_frequencies {
        let mut sorted_ngrams: Vec<(String, usize)> = freq_map.iter()
            .map(|(ngram, count)| (ngram.clone(), *count))
            .collect();
        sorted_ngrams.sort_by(|a, b| b.1.cmp(&a.1));
        sorted_ngrams.truncate(100); // Keep top 100 for each n-gram size
        ngram_output.insert(*n, sorted_ngrams);
    }
    
    // Extract common phrases (interesting 3-grams and 4-grams)
    let mut common_phrases = Vec::new();
    if let Some(trigrams) = ngram_output.get(&3) {
        common_phrases.extend(trigrams.iter().take(20).cloned());
    }
    if let Some(fourgrams) = ngram_output.get(&4) {
        common_phrases.extend(fourgrams.iter().take(20).cloned());
    }
    common_phrases.sort_by(|a, b| b.1.cmp(&a.1));
    common_phrases.truncate(25);
    
    // Calculate global sentiment score
    let sentiment_scores = calculate_sentiment_scores(&all_text);
    
    // Calculate readability scores based on aggregated data
    let readability_scores = calculate_readability_scores(
        total_words,
        sentence_lengths.len(),
        file_results.iter().map(|r| r.syllable_count).sum(),
        &sentence_lengths,
    );
    
    // Extract sentiment score as a separate field for easier access
    let overall_sentiment = sentiment_scores.get("Overall Sentiment").cloned();
    
    // Create a simple parts-of-speech map
    // This is a simplified version; a real implementation would use a POS tagger
    let mut pos_counts = HashMap::new();
    pos_counts.insert("NOUN".to_string(), (total_words as f64 * 0.28) as usize);
    pos_counts.insert("VERB".to_string(), (total_words as f64 * 0.20) as usize);
    pos_counts.insert("ADJ".to_string(), (total_words as f64 * 0.12) as usize);
    pos_counts.insert("ADV".to_string(), (total_words as f64 * 0.07) as usize);
    pos_counts.insert("PRON".to_string(), (total_words as f64 * 0.08) as usize);
    pos_counts.insert("DET".to_string(), (total_words as f64 * 0.09) as usize);
    pos_counts.insert("ADP".to_string(), (total_words as f64 * 0.10) as usize);
    pos_counts.insert("CONJ".to_string(), (total_words as f64 * 0.04) as usize);
    pos_counts.insert("PART".to_string(), (total_words as f64 * 0.02) as usize);
    
    // Get the sentence count before we move sentence_lengths
    let sentence_count = sentence_lengths.len();
    
    Ok(AnalysisResults {
        files_processed: files.len(),
        total_words,
        unique_words: word_frequencies.len(),
        avg_words_per_file: total_words as f64 / files.len() as f64,
        top_words,
        word_frequencies,
        character_frequencies,
        sentence_lengths,
        readability_scores,
        sentiment_scores,
        sentiment_score: overall_sentiment,
        sentence_count,
        ngram_frequencies: ngram_output,
        common_phrases,
        file_stats,
        pos_counts,
    })
}

struct FileAnalysisResult {
    filename: String,
    word_count: usize,
    word_frequencies: HashMap<String, usize>,
    character_frequencies: HashMap<char, usize>,
    sentence_lengths: Vec<usize>,
    syllable_count: usize,
    avg_word_length: f64,
    longest_word: String,
    full_text: String,
    sentiment_score: f64,
    ngram_frequencies: HashMap<usize, HashMap<String, usize>>,
}

fn analyze_file(path: &Path) -> Result<FileAnalysisResult> {
    let file = File::open(path).context(format!("Failed to open file: {}", path.display()))?;
    let reader = BufReader::new(file);
    
    let mut word_frequencies = HashMap::new();
    let mut character_frequencies = HashMap::new();
    let mut sentence_lengths = Vec::new();
    let mut current_sentence_words = 0;
    let mut total_characters = 0;
    let mut syllable_count = 0;
    let mut longest_word = String::new();
    let mut full_text = String::new();
    
    // N-gram maps (for 2-grams, 3-grams, and 4-grams)
    let mut ngram_frequencies: HashMap<usize, HashMap<String, usize>> = HashMap::new();
    for n in 2..=4 {
        ngram_frequencies.insert(n, HashMap::new());
    }
    
    // Store words in order for n-gram analysis
    let mut words_in_order = Vec::new();
    
    // Regex for word extraction
    let word_regex = Regex::new(r"\b[a-zA-Z0-9']+\b").unwrap();
    // Regex for sentence endings
    let sentence_end_regex = Regex::new(r"[.!?]+").unwrap();
    
    for line in reader.lines() {
        let line = line.context("Failed to read line")?;
        full_text.push_str(&line);
        full_text.push('\n');
        
        // Count characters
        for ch in line.chars() {
            if !ch.is_whitespace() {
                *character_frequencies.entry(ch).or_insert(0) += 1;
                total_characters += 1;
            }
        }
        
        // Count words and check for sentence endings
        for (i, word) in word_regex.find_iter(&line).enumerate() {
            let word_str = word.as_str().to_lowercase();
            *word_frequencies.entry(word_str.clone()).or_insert(0) += 1;
            
            // Add word to ordered list for n-gram analysis
            words_in_order.push(word_str.clone());
            
            // Track longest word
            if word_str.len() > longest_word.len() {
                longest_word = word_str.clone();
            }
            
            // Count syllables (naive approach)
            syllable_count += estimate_syllables(&word_str);
            
            current_sentence_words += 1;
            
            // Check if we're at a sentence end
            if i + 1 < line.len() && sentence_end_regex.is_match(&line[word.end()..]) {
                if current_sentence_words > 0 {
                    sentence_lengths.push(current_sentence_words);
                    current_sentence_words = 0;
                }
            }
        }
    }
    
    // Handle any remaining sentence
    if current_sentence_words > 0 {
        sentence_lengths.push(current_sentence_words);
    }
    
    // Generate n-grams
    generate_ngrams(&words_in_order, &mut ngram_frequencies);
    
    let word_count = word_frequencies.values().sum();
    let avg_word_length = if word_count > 0 {
        total_characters as f64 / word_count as f64
    } else {
        0.0
    };
    
    // Calculate sentiment score for this file
    let sentiment_score = calculate_simple_sentiment(&full_text);
    
    Ok(FileAnalysisResult {
        filename: path.file_name().unwrap_or_default().to_string_lossy().to_string(),
        word_count,
        word_frequencies,
        character_frequencies,
        sentence_lengths,
        syllable_count,
        avg_word_length,
        longest_word,
        full_text,
        sentiment_score,
        ngram_frequencies,
    })
}

fn generate_ngrams(words: &[String], ngram_frequencies: &mut HashMap<usize, HashMap<String, usize>>) {
    for n in 2..=4 {
        if words.len() < n {
            continue;
        }
        
        let ngram_map = ngram_frequencies.get_mut(&n).unwrap();
        
        for i in 0..=(words.len() - n) {
            let ngram = words[i..i+n].join(" ");
            *ngram_map.entry(ngram).or_insert(0) += 1;
        }
    }
}

fn collect_files(paths: &[PathBuf], recursive: bool, pattern: Option<&str>) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    let pattern_regex = pattern.map(|p| Regex::new(p).context("Invalid regex pattern")).transpose()?;
    
    for path in paths {
        if path.is_file() {
            if should_include_file(path, &pattern_regex) {
                files.push(path.clone());
            }
        } else if path.is_dir() {
            let walker = if recursive {
                WalkDir::new(path).into_iter()
            } else {
                WalkDir::new(path).max_depth(1).into_iter()
            };
            
            for entry in walker.filter_map(Result::ok) {
                let entry_path = entry.path();
                if entry_path.is_file() && should_include_file(entry_path, &pattern_regex) {
                    files.push(entry_path.to_path_buf());
                }
            }
        }
    }
    
    Ok(files)
}

fn should_include_file(path: &Path, pattern: &Option<Regex>) -> bool {
    if let Some(ext) = path.extension() {
        // By default, only analyze text-like files
        let is_text = matches!(ext.to_string_lossy().as_ref(), 
            "txt" | "md" | "rs" | "js" | "py" | "java" | "c" | "cpp" | "h" | "hpp" | "html" | "css" |
            "go" | "rb" | "pl" | "php" | "ts" | "json" | "yaml" | "yml" | "toml" | "csv" | "log");
        
        if let Some(pattern) = pattern {
            // If pattern specified, apply it to the filename
            let filename = path.file_name().unwrap_or_default().to_string_lossy();
            pattern.is_match(&filename) && is_text
        } else {
            is_text
        }
    } else {
        false
    }
}

fn estimate_syllables(word: &str) -> usize {
    // A more sophisticated syllable counter
    let word = word.to_lowercase();
    
    // Special cases
    if word.len() <= 3 {
        return 1;
    }
    
    let mut syllables = 0;
    let mut prev_is_vowel = false;
    
    // Count vowel groups
    for c in word.chars() {
        let is_vowel = "aeiouy".contains(c);
        if is_vowel && !prev_is_vowel {
            syllables += 1;
        }
        prev_is_vowel = is_vowel;
    }
    
    // Apply special rules
    if word.ends_with('e') && syllables > 1 && !word.ends_with("le") {
        syllables -= 1;
    }
    
    // Words ending with "y" usually have a syllable there
    if word.ends_with('y') && syllables == 0 {
        syllables = 1;
    }
    
    // Common suffixes
    if word.ends_with("ion") || word.ends_with("ious") || word.ends_with("ia") || word.ends_with("eal") {
        syllables += 1;
    }
    
    // Every word has at least one syllable
    syllables.max(1)
}

fn calculate_readability_scores(total_words: usize, total_sentences: usize, total_syllables: usize, sentence_lengths: &[usize]) -> HashMap<String, f64> {
    let mut scores = HashMap::new();
    
    if total_sentences > 0 && total_words > 0 {
        // Average words per sentence
        let avg_words_per_sentence = total_words as f64 / total_sentences as f64;
        scores.insert("Words per Sentence".to_string(), avg_words_per_sentence);
        
        // Flesch-Kincaid Grade Level
        if total_syllables > 0 {
            let fk_grade = 0.39 * avg_words_per_sentence + 11.8 * (total_syllables as f64 / total_words as f64) - 15.59;
            scores.insert("Flesch-Kincaid Grade".to_string(), fk_grade);
            
            // Flesch Reading Ease
            let reading_ease = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * (total_syllables as f64 / total_words as f64);
            scores.insert("Flesch Reading Ease".to_string(), reading_ease);
            
            // Calculate reading level category and add the score directly
            let _reading_level = match reading_ease as i32 {
                90..=100 => "Very Easy",
                80..=89 => "Easy",
                70..=79 => "Fairly Easy",
                60..=69 => "Standard",
                50..=59 => "Fairly Difficult",
                30..=49 => "Difficult",
                _ => "Very Difficult",
            };
            scores.insert("Reading Level".to_string(), reading_ease); // We'll display the category in the UI
        }
        
        // Gunning Fog Index
        let complex_words = total_words / 10; // Simplified assumption
        let fog_index = 0.4 * (avg_words_per_sentence + 100.0 * (complex_words as f64 / total_words as f64));
        scores.insert("Gunning Fog Index".to_string(), fog_index);
        
        // Coleman-Liau Index (approximation)
        let avg_letters_per_100_words = (total_syllables as f64 * 1.5) * 100.0 / total_words as f64;
        let avg_sentences_per_100_words = total_sentences as f64 * 100.0 / total_words as f64;
        let coleman_liau = 0.0588 * avg_letters_per_100_words - 0.296 * avg_sentences_per_100_words - 15.8;
        scores.insert("Coleman-Liau Index".to_string(), coleman_liau);
        
        // Sentence complexity (standard deviation)
        if sentence_lengths.len() > 1 {
            let mean = avg_words_per_sentence;
            let variance: f64 = sentence_lengths.iter()
                .map(|&len| (len as f64 - mean).powi(2))
                .sum::<f64>() / sentence_lengths.len() as f64;
            let std_dev = variance.sqrt();
            scores.insert("Sentence Complexity".to_string(), std_dev);
        }
    }
    
    scores
}

fn calculate_simple_sentiment(text: &str) -> f64 {
    // This is a very simplified sentiment analysis
    // In a real implementation, you would use a proper NLP library
    
    let positive_words = [
        "good", "great", "excellent", "wonderful", "happy", "positive", "best", "love", "likes",
        "advantage", "benefit", "effective", "efficient", "improvement", "innovative", "reliable",
        "secure", "speed", "success", "superior", "valuable",
    ];
    
    let negative_words = [
        "bad", "poor", "terrible", "horrible", "sad", "negative", "worst", "hate", "dislikes",
        "disadvantage", "failure", "ineffective", "inefficient", "problem", "slow", "unstable",
        "unsecure", "vulnerability", "failure", "inferior", "worthless",
    ];
    
    let word_regex = Regex::new(r"\b[a-zA-Z']+\b").unwrap();
    let words: Vec<_> = word_regex.find_iter(text)
        .map(|m| m.as_str().to_lowercase())
        .collect();
    
    let total_words = words.len() as f64;
    if total_words == 0.0 {
        return 0.0;
    }
    
    let mut positive_count = 0;
    let mut negative_count = 0;
    
    for word in words {
        if positive_words.contains(&word.as_str()) {
            positive_count += 1;
        } else if negative_words.contains(&word.as_str()) {
            negative_count += 1;
        }
    }
    
    // Calculate sentiment score (-1.0 to 1.0)
    let sentiment_words = positive_count + negative_count;
    if sentiment_words == 0 {
        return 0.0;
    }
    
    (positive_count as f64 - negative_count as f64) / sentiment_words as f64
}

fn calculate_sentiment_scores(text: &str) -> HashMap<String, f64> {
    let mut scores = HashMap::new();
    
    let base_sentiment = calculate_simple_sentiment(text);
    scores.insert("Overall Sentiment".to_string(), base_sentiment);
    
    // Sentiment category
    let sentiment_category = match base_sentiment {
        s if s >= 0.6 => 5.0,  // Very Positive
        s if s >= 0.2 => 4.0,  // Positive
        s if s > -0.2 => 3.0,  // Neutral
        s if s > -0.6 => 2.0,  // Negative
        _ => 1.0,              // Very Negative
    };
    scores.insert("Sentiment Category".to_string(), sentiment_category);
    
    scores
} 