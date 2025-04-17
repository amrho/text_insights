use anyhow::{Context, Result};
use serde_json;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::analyzer::AnalysisResults;

#[derive(Debug, Serialize, Deserialize)]
pub struct ComparisonData {
    pub documents: Vec<DocumentSummary>,
    pub similarity_matrix: Vec<Vec<f64>>,
    pub common_words: Vec<String>,
    pub distinctive_words: HashMap<String, Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DocumentSummary {
    pub name: String,
    pub word_count: usize,
    pub unique_words: usize,
    pub readability: Option<f64>,
    pub sentiment: Option<f64>,
    pub top_words: Vec<(String, usize)>,
}

pub fn export_to_json<T: serde::Serialize>(data: &T, path: &Path) -> Result<()> {
    let file = File::create(path).context("Failed to create output JSON file")?;
    let writer = BufWriter::new(file);
    
    serde_json::to_writer_pretty(writer, data)
        .context("Failed to serialize data to JSON")
}

pub fn import_from_json(path: &Path) -> Result<AnalysisResults> {
    let file = File::open(path).context("Failed to open JSON file")?;
    let reader = BufReader::new(file);
    
    serde_json::from_reader(reader)
        .context("Failed to deserialize analysis results from JSON")
}

pub fn create_comparison_data(results: &[AnalysisResults]) -> ComparisonData {
    let mut documents = Vec::new();
    let mut similarity_matrix = Vec::new();
    let mut all_words = HashSet::new();
    let mut word_frequencies_by_doc = Vec::new();
    
    // First pass: extract document summaries and collect all words
    for result in results {
        // Extract document name
        let name = if let Some(first_file) = result.file_stats.first() {
            first_file.filename.clone()
        } else {
            format!("Document {}", documents.len() + 1)
        };
        
        // Collect words from this document
        let doc_words: HashSet<_> = result.word_frequencies.keys().cloned().collect();
        all_words.extend(doc_words);
        
        // Store word frequencies for later processing
        word_frequencies_by_doc.push(&result.word_frequencies);
        
        // Extract readability and sentiment scores
        let readability = result.readability_scores.get("Flesch Reading Ease").copied();
        let sentiment = result.sentiment_scores.get("Overall Sentiment").copied();
        
        // Create document summary
        documents.push(DocumentSummary {
            name,
            word_count: result.total_words,
            unique_words: result.unique_words,
            readability,
            sentiment,
            top_words: result.top_words.iter().take(10).cloned().collect(),
        });
    }
    
    // Second pass: calculate similarity matrix
    for i in 0..results.len() {
        let mut row = Vec::new();
        let words_a: HashSet<_> = results[i].word_frequencies.keys().collect();
        
        for j in 0..results.len() {
            let words_b: HashSet<_> = results[j].word_frequencies.keys().collect();
            
            let intersection = words_a.intersection(&words_b).count();
            let union = words_a.union(&words_b).count();
            
            let similarity = if union == 0 {
                0.0
            } else {
                intersection as f64 / union as f64
            };
            
            row.push(similarity);
        }
        
        similarity_matrix.push(row);
    }
    
    // Find common words across all documents (appear in at least 80% of documents)
    let threshold = (results.len() * 4) / 5; // 80% of documents
    let common_words: Vec<_> = all_words
        .iter()
        .filter(|word| {
            let count = word_frequencies_by_doc.iter()
                .filter(|freqs| freqs.contains_key(*word))
                .count();
            count >= threshold
        })
        .cloned()
        .collect();
    
    // Find distinctive words for each document
    let mut distinctive_words = HashMap::new();
    
    for (i, result) in results.iter().enumerate() {
        let mut distinctive = Vec::new();
        
        for (word, &count) in &result.word_frequencies {
            if count <= 1 {
                continue; // Ignore words that appear only once
            }
            
            // Check if word is rare in other documents
            let mut is_distinctive = true;
            for (j, other_result) in results.iter().enumerate() {
                if i == j {
                    continue;
                }
                
                if let Some(&other_count) = other_result.word_frequencies.get(word) {
                    if other_count >= count / 2 {
                        is_distinctive = false;
                        break;
                    }
                }
            }
            
            if is_distinctive {
                distinctive.push(word.clone());
            }
        }
        
        // Sort by frequency and take top 10
        distinctive.sort_by(|a, b| {
            let count_a = result.word_frequencies.get(a).unwrap_or(&0);
            let count_b = result.word_frequencies.get(b).unwrap_or(&0);
            count_b.cmp(count_a)
        });
        
        distinctive_words.insert(documents[i].name.clone(), distinctive.into_iter().take(10).collect());
    }
    
    ComparisonData {
        documents,
        similarity_matrix,
        common_words,
        distinctive_words,
    }
}

// Function to get a safe filename from a string
pub fn sanitize_filename(s: &str) -> String {
    // Replace characters that might be problematic in filenames
    let mut result = s.to_string();
    for c in &['/', '\\', ':', '*', '?', '"', '<', '>', '|'] {
        result = result.replace(*c, "_");
    }
    
    // Trim whitespace and truncate if too long
    let result = result.trim();
    if result.len() > 50 {
        result[..50].to_string()
    } else {
        result.to_string()
    }
}

// Generate a human-readable summary of given text
pub fn generate_text_summary(text: &str, max_length: usize) -> String {
    if text.len() <= max_length {
        return text.to_string();
    }
    
    // Find a suitable breakpoint (preferably at a sentence end)
    let truncated = &text[..max_length];
    let breakpoints = [". ", "! ", "? ", "; ", ", ", " "];
    
    for &bp in &breakpoints {
        if let Some(pos) = truncated.rfind(bp) {
            return format!("{}...", &text[..pos + 1]);
        }
    }
    
    // If no good breakpoint found, just truncate with ellipsis
    format!("{}...", truncated)
}

// Calculate the date range from a collection of timestamps
pub fn format_date_range(timestamps: &[i64]) -> String {
    use std::time::{Duration, UNIX_EPOCH};
    
    if timestamps.is_empty() {
        return "No data".to_string();
    }
    
    // Find min and max timestamps
    let min_ts = timestamps.iter().min().unwrap();
    let max_ts = timestamps.iter().max().unwrap();
    
    // Convert to date strings
    let min_date = UNIX_EPOCH + Duration::from_secs(*min_ts as u64);
    let max_date = UNIX_EPOCH + Duration::from_secs(*max_ts as u64);
    
    let format_date = |date| {
        let dt = chrono::DateTime::<chrono::Utc>::from(date);
        dt.format("%Y-%m-%d").to_string()
    };
    
    if min_ts == max_ts {
        format_date(min_date)
    } else {
        format!("{} to {}", format_date(min_date), format_date(max_date))
    }
}

// Return a pluralized form of a word based on count
pub fn pluralize(word: &str, count: usize) -> String {
    if count == 1 {
        word.to_string()
    } else {
        // Basic English pluralization
        if word.ends_with('s') || word.ends_with('x') || word.ends_with('z') ||
           word.ends_with("ch") || word.ends_with("sh") {
            format!("{}es", word)
        } else if word.ends_with('y') && !word.ends_with("ay") && 
                 !word.ends_with("ey") && !word.ends_with("iy") && 
                 !word.ends_with("oy") && !word.ends_with("uy") {
            format!("{}ies", &word[..word.len()-1])
        } else {
            format!("{}s", word)
        }
    }
} 