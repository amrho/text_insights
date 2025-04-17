use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::io::{self, Write};
use std::collections::HashSet;
use colored::*;

mod analyzer;
mod visualizer;
mod utils;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Analyze text files and generate insights
    Analyze {
        /// Files or directories to analyze
        #[arg(required = true)]
        paths: Vec<PathBuf>,

        /// Recursively search directories
        #[arg(short, long)]
        recursive: bool,

        /// Output results as JSON to specified file
        #[arg(short, long)]
        output_json: Option<PathBuf>,

        /// Filter to only include files matching pattern
        #[arg(short, long)]
        pattern: Option<String>,

        /// Run in detailed mode with more analysis
        #[arg(short, long)]
        detailed: bool,
    },
    /// Generate visualizations from analysis data
    Visualize {
        /// Input JSON file with analysis results
        #[arg(short, long)]
        input: PathBuf,

        /// Type of visualization to generate
        #[arg(short, long, default_value = "word-frequency")]
        r#type: String,

        /// Output file for the visualization
        #[arg(short, long)]
        output: PathBuf,

        /// Limit results to top N items
        #[arg(short, long, default_value = "50")]
        limit: usize,
    },
    /// Compare multiple text files or directories
    Compare {
        /// Files or directories to compare (2-5 items)
        #[arg(required = true)]
        paths: Vec<PathBuf>,
        
        /// Output comparison results as JSON
        #[arg(short, long)]
        output_json: Option<PathBuf>,
        
        /// Generate comparison chart
        #[arg(short, long)]
        chart: Option<PathBuf>,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Analyze {
            paths,
            recursive,
            output_json,
            pattern,
            detailed,
        } => {
            let results = analyzer::analyze_paths(paths, *recursive, pattern.as_deref())
                .context("Failed to analyze text")?;
            
            // Print summary to console
            print_analysis_summary(&results, *detailed);
            
            // Export to JSON if requested
            if let Some(json_path) = output_json {
                utils::export_to_json(&results, json_path)
                    .context("Failed to export results to JSON")?;
                println!("Results exported to {}", json_path.display());
            }
            
            Ok(())
        }
        Commands::Visualize {
            input,
            r#type,
            output,
            limit,
        } => {
            let results = utils::import_from_json(input)
                .context("Failed to import analysis results")?;
            
            visualizer::generate_visualization(&results, r#type, output, *limit)
                .context("Failed to generate visualization")?;
            
            println!("Visualization saved to {}", output.display());
            Ok(())
        }
        Commands::Compare {
            paths,
            output_json,
            chart,
        } => {
            if paths.len() < 2 || paths.len() > 5 {
                return Err(anyhow::anyhow!("Compare requires 2-5 paths to compare"));
            }
            
            let mut results = Vec::new();
            let pb = indicatif::ProgressBar::new(paths.len() as u64);
            pb.set_style(indicatif::ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} items ({eta})")
                .unwrap()
                .progress_chars("#>-"));
            
            for path in paths {
                let result = analyzer::analyze_paths(&[path.clone()], true, None)
                    .context(format!("Failed to analyze {}", path.display()))?;
                results.push(result);
                pb.inc(1);
            }
            pb.finish_with_message("Comparison complete!");
            
            // Print comparison results
            print_comparison_results(&results);
            
            // Export to JSON if requested
            if let Some(json_path) = output_json {
                let comparison = utils::create_comparison_data(&results);
                utils::export_to_json(&comparison, json_path)
                    .context("Failed to export comparison results to JSON")?;
                println!("Comparison results exported to {}", json_path.display());
            }
            
            // Generate chart if requested
            if let Some(chart_path) = chart {
                visualizer::generate_comparison_chart(&results, &chart_path)
                    .context("Failed to generate comparison chart")?;
                println!("Comparison chart saved to {}", chart_path.display());
            }
            
            Ok(())
        }
    }
}

fn print_analysis_summary(results: &analyzer::AnalysisResults, detailed: bool) {
    println!("\n{}", "=== Text Insights Analysis Summary ===".bold().green());
    println!("Files processed: {}", results.files_processed);
    println!("Total words: {}", results.total_words);
    println!("Unique words: {}", results.unique_words);
    println!("Average words per file: {:.2}", results.avg_words_per_file);
    
    // Print sentiment information
    if let Some(sentiment) = results.sentiment_scores.get("Overall Sentiment") {
        let sentiment_desc = match *sentiment {
            s if s >= 0.6 => "Very Positive".bright_green(),
            s if s >= 0.2 => "Positive".green(),
            s if s > -0.2 => "Neutral".yellow(),
            s if s > -0.6 => "Negative".red(),
            _ => "Very Negative".bright_red(),
        };
        
        println!("\nSentiment Analysis: {} ({:.2})", sentiment_desc, sentiment);
    }
    
    if !results.top_words.is_empty() {
        println!("\n{}", "Top 10 most common words:".bold());
        for (i, (word, count)) in results.top_words.iter().take(10).enumerate() {
            println!("  {}. {} ({})", i + 1, word, count);
        }
    }
    
    if !results.readability_scores.is_empty() {
        println!("\n{}", "Readability scores:".bold());
        
        // Reading level
        if let Some(score) = results.readability_scores.get("Flesch Reading Ease") {
            let level = match *score as i32 {
                90..=100 => "Very Easy".bright_green(),
                80..=89 => "Easy".green(),
                70..=79 => "Fairly Easy".yellow(),
                60..=69 => "Standard".normal(),
                50..=59 => "Fairly Difficult".yellow(),
                30..=49 => "Difficult".red(),
                _ => "Very Difficult".bright_red(),
            };
            println!("  Reading Level: {}", level);
        }
        
        // Print selected readability metrics
        let important_metrics = ["Flesch-Kincaid Grade", "Flesch Reading Ease", "Coleman-Liau Index"];
        for metric in important_metrics {
            if let Some(score) = results.readability_scores.get(metric) {
                println!("  {}: {:.2}", metric, score);
            }
        }
    }
    
    // Only print these sections in detailed mode
    if detailed {
        // Print common phrases if available
        if !results.common_phrases.is_empty() {
            println!("\n{}", "Common phrases:".bold());
            for (i, (phrase, count)) in results.common_phrases.iter().take(5).enumerate() {
                println!("  {}. \"{}\" ({})", i + 1, phrase, count);
            }
        }
        
        // Show file statistics
        if !results.file_stats.is_empty() && results.file_stats.len() <= 10 {
            println!("\n{}", "File statistics:".bold());
            for stats in &results.file_stats {
                println!("  {}: {} words, {:.2} avg word length", 
                    stats.filename, stats.word_count, stats.avg_word_length);
            }
        }
        
        // Ask user if they want to see more detailed information
        if results.top_words.len() > 10 {
            print!("\nShow more details? [y/N]: ");
            io::stdout().flush().unwrap();
            
            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            
            if input.trim().to_lowercase() == "y" {
                // Show more common words
                println!("\n{}", "Top 20 most common words:".bold());
                for (i, (word, count)) in results.top_words.iter().take(20).enumerate() {
                    println!("  {}. {} ({})", i + 1, word, count);
                }
                
                // Show bigrams
                if let Some(bigrams) = results.ngram_frequencies.get(&2) {
                    println!("\n{}", "Common word pairs:".bold());
                    for (i, (phrase, count)) in bigrams.iter().take(10).enumerate() {
                        println!("  {}. \"{}\" ({})", i + 1, phrase, count);
                    }
                }
                
                // Show additional readability metrics
                println!("\n{}", "All readability metrics:".bold());
                for (metric, score) in &results.readability_scores {
                    println!("  {}: {:.2}", metric, score);
                }
            }
        }
    }
}

fn print_comparison_results(results: &[analyzer::AnalysisResults]) {
    println!("\n{}", "=== Text Comparison Results ===".bold().blue());
    
    // Create a table header
    let mut headers = vec!["Metric".to_string()];
    for (i, result) in results.iter().enumerate() {
        let name = if let Some(first_file) = result.file_stats.first() {
            first_file.filename.clone()
        } else {
            format!("Document {}", i + 1)
        };
        headers.push(name);
    }
    
    // Print metrics table
    print_table(&[
        headers.clone(),
        create_table_row("Total Words", results, |r| r.total_words.to_string()),
        create_table_row("Unique Words", results, |r| r.unique_words.to_string()),
        create_table_row("Reading Ease", results, |r| {
            r.readability_scores.get("Flesch Reading Ease")
                .map(|s| format!("{:.1}", s))
                .unwrap_or_else(|| "-".to_string())
        }),
        create_table_row("Sentiment", results, |r| {
            r.sentiment_scores.get("Overall Sentiment")
                .map(|s| format!("{:.2}", s))
                .unwrap_or_else(|| "-".to_string())
        }),
    ]);
    
    // Print vocabulary overlap
    println!("\n{}", "Vocabulary Similarity:".bold());
    for i in 0..results.len() {
        for j in i+1..results.len() {
            let overlap = calculate_vocabulary_overlap(&results[i], &results[j]);
            println!("  {} & {}: {:.1}% similar", headers[i+1], headers[j+1], overlap * 100.0);
        }
    }
    
    // Print unique words to each document
    println!("\n{}", "Most distinctive words per document:".bold());
    for (i, result) in results.iter().enumerate() {
        let mut distinctive_words = get_distinctive_words(result, results, i);
        distinctive_words.sort_by(|a, b| b.1.cmp(&a.1));
        
        println!("  {}: {}", headers[i+1], 
            distinctive_words.iter().take(5).map(|(w, _)| w.as_str()).collect::<Vec<_>>().join(", "));
    }
}

fn create_table_row<F>(metric: &str, results: &[analyzer::AnalysisResults], extractor: F) -> Vec<String>
where F: Fn(&analyzer::AnalysisResults) -> String {
    let mut row = vec![metric.to_string()];
    for result in results {
        row.push(extractor(result));
    }
    row
}

fn print_table(rows: &[Vec<String>]) {
    if rows.is_empty() || rows[0].is_empty() {
        return;
    }
    
    // Calculate column widths
    let cols = rows[0].len();
    let mut col_widths = vec![0; cols];
    
    for row in rows {
        for (i, cell) in row.iter().enumerate() {
            col_widths[i] = col_widths[i].max(cell.len());
        }
    }
    
    // Print header row
    if !rows.is_empty() {
        let header = &rows[0];
        print!("+");
        for (_i, width) in col_widths.iter().enumerate() {
            print!("{}", "-".repeat(width + 2));
            print!("+");
        }
        println!();
        
        print!("|");
        for (i, cell) in header.iter().enumerate() {
            print!(" {}{} |", cell, " ".repeat(col_widths[i] - cell.len()));
        }
        println!();
        
        print!("+");
        for (_i, width) in col_widths.iter().enumerate() {
            print!("{}", "=".repeat(width + 2));
            print!("+");
        }
        println!();
    }
    
    // Print data rows
    for row in &rows[1..] {
        print!("|");
        for (i, cell) in row.iter().enumerate() {
            print!(" {}{} |", cell, " ".repeat(col_widths[i] - cell.len()));
        }
        println!();
    }
    
    // Print bottom border
    print!("+");
    for (_i, width) in col_widths.iter().enumerate() {
        print!("{}", "-".repeat(width + 2));
        print!("+");
    }
    println!();
}

fn calculate_vocabulary_overlap(a: &analyzer::AnalysisResults, b: &analyzer::AnalysisResults) -> f64 {
    let words_a: HashSet<_> = a.word_frequencies.keys().collect();
    let words_b: HashSet<_> = b.word_frequencies.keys().collect();
    
    let intersection = words_a.intersection(&words_b).count();
    let union = words_a.union(&words_b).count();
    
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}

fn get_distinctive_words(document: &analyzer::AnalysisResults, all_docs: &[analyzer::AnalysisResults], doc_index: usize) -> Vec<(String, usize)> {
    let mut distinctive_words = Vec::new();
    
    for (word, &count) in document.word_frequencies.iter().filter(|(_, &count)| count > 1) {
        let mut distinctiveness = count;
        
        // Check if word appears in other documents
        for (i, other_doc) in all_docs.iter().enumerate() {
            if i == doc_index {
                continue;
            }
            
            if let Some(&other_count) = other_doc.word_frequencies.get(word) {
                distinctiveness = distinctiveness.saturating_sub(other_count);
            }
        }
        
        if distinctiveness > 0 {
            distinctive_words.push((word.clone(), distinctiveness));
        }
    }
    
    distinctive_words
}
